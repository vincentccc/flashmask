/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
                __FILE__ + ":" +                         \
                ::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)

static thread_local std::unique_ptr<char[]> flash_attn_err_msg;

void flash_attn_set_error(const char *msg) {
  if (msg == nullptr || *msg == '\0') {
    msg = "unknown error";
  }

  auto n = strlen(msg);
  std::unique_ptr<char[]> new_err_msg(new char[n+1]);
  std::strcpy(new_err_msg.get(), msg);
  flash_attn_err_msg = std::move(new_err_msg);
}

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define FLASHATTNLIB_BEGIN_FUNC try {
#define FLASHATTNLIB_END_FUNC } catch (::std::exception &__e) { flash_attn_set_error(__e.what()); return false; } catch (...) { flash_attn_set_error(nullptr); return false; }

#define CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                     \
      auto dprops = at::cuda::getCurrentDeviceProperties();              \
      const bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;     \
      const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;     \
      ASSERT_CHECK(is_sm8x || is_sm90);                                  \
      ASSERT_CHECK(batch_size > 0);                                      \
      ASSERT_CHECK(head_size % 8 == 0);                                  \
      ASSERT_CHECK(head_size <= 256);                                    \
      ASSERT_CHECK(num_heads % num_heads_k == 0);                        \
      if (attn_mask) {                                                   \
          ASSERT_CHECK(mask_dims[0] == batch_size);                      \
          ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);  \
          ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == __seqlen_q); \
          ASSERT_CHECK(mask_dims[3] == __seqlen_k);                      \
      }

#define CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                         \
      const bool is_sm80 = dprops->major == 8 && dprops->minor == 0;                       \
      if (head_size > 192) {                                                               \
          /* FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800 */ \
          ASSERT_CHECK(is_sm80 || is_sm90);                                                \
      }

#define CHECK_CALC_REDUCED_SCORES_EXECTUABLE(__seqlen_q, __seqlen_k) \
      const void * attn_mask = nullptr;                              \
      const int64_t * mask_dims = nullptr;                           \
      CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)

void set_params_fprop_strided(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void * const q,
                      void * const k,
                      void * const v,
                      void * const out,
                      void * const cu_seqlens_q_d,
                      void * const cu_seqlens_k_d,
                      void * const p_d,
                      void * const softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      bool is_causal,
                      bool is_bf16,
                      const int q_row_stride,
                      const int k_row_stride,
                      const int v_row_stride,
                      const int q_head_stride,
                      const int k_head_stride,
                      const int v_head_stride,
                      const int o_row_stride,
                      const int o_head_stride,
                      const int q_batch_stride,
                      const int k_batch_stride,
                      const int v_batch_stride,
                      const int o_batch_stride,
                      bool varlen_padded_input = false,
                      void * attn_mask = nullptr,
                      void * flashmask_downstart_ptr = nullptr,
                      void * flashmask_upend_ptr = nullptr,
                      void * flashmask_downend_ptr = nullptr,
                      void * flashmask_upstart_ptr = nullptr,
                      void * flashmask_maxmin_ptr = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0) {
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;
    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_ptr = out;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;
    params.varlen_padded_input = varlen_padded_input;

    if (cu_seqlens_q_d == nullptr ||  params.varlen_padded_input) {
        params.q_batch_stride = q_batch_stride;
        params.k_batch_stride = k_batch_stride;
        params.v_batch_stride = v_batch_stride;
        params.o_batch_stride = o_batch_stride;
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // attn mask
    params.attn_mask_ptr = attn_mask;
    params.mask_head_mod_size = mask_head_mod_size;
    params.mask_seq_q_mod_size = mask_seq_q_mod_size;

    // sparse mask row index
    params.flashmask_downstart_ptr = flashmask_downstart_ptr;
    params.flashmask_upend_ptr = flashmask_upend_ptr;
    params.flashmask_downend_ptr = flashmask_downend_ptr;
    params.flashmask_upstart_ptr = flashmask_upstart_ptr;
    params.flashmask_maxmin_ptr = static_cast<int*>(flashmask_maxmin_ptr);
    params.enable_mask_bypass = true;
    if(flashmask_downstart_ptr != nullptr || flashmask_upend_ptr != nullptr) {
        params.h_sparsemask = mask_head_mod_size;
        params.h_h_sparsemask_ratio = h / mask_head_mod_size;
        if (params.enable_mask_bypass){
            ASSERT_CHECK(params.flashmask_maxmin_ptr != nullptr);
        }
    }

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.unscale_softmax = softmax_unscale;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    ASSERT_CHECK(p_dropout < 1.f);

    params.is_causal = is_causal;
}

void set_params_dgrad_strided(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void * const q,
                      void * const k,
                      void * const v,
                      void * const out,
                      void * const dout,
                      void * const dq,
                      void * const dk,
                      void * const dv,
                      void * const cu_seqlens_q_d,
                      void * const cu_seqlens_k_d,
                      void * const dq_accum_d,
                      void * const dk_accum_d,
                      void * const dv_accum_d,
                      void * const softmax_lse_d,
                      void * const dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      bool is_causal,
                      bool is_bf16,
                      const int q_row_stride,
                      const int k_row_stride,
                      const int v_row_stride,
                      const int q_head_stride,
                      const int k_head_stride,
                      const int v_head_stride,
                      const int o_row_stride,
                      const int o_head_stride,
                      const int q_batch_stride,
                      const int k_batch_stride,
                      const int v_batch_stride,
                      const int o_batch_stride,
                      const int dq_row_stride,
                      const int dk_row_stride,
                      const int dv_row_stride,
                      const int dq_head_stride,
                      const int dk_head_stride,
                      const int dv_head_stride,
                      const int do_row_stride,
                      const int do_head_stride,
                      const int dq_batch_stride,
                      const int dk_batch_stride,
                      const int dv_batch_stride,
                      const int do_batch_stride,
                      const bool varlen_padded_input = false,
                      const int num_splits = 0,
                      void * attn_mask = nullptr,
                      void * flashmask_downstart_ptr = nullptr,
                      void * flashmask_upend_ptr = nullptr,
                      void * flashmask_downend_ptr = nullptr,
                      void * flashmask_upstart_ptr = nullptr,
                      void * flashmask_maxmin_ptr = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0) {

    set_params_fprop_strided(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     q_row_stride,k_row_stride,v_row_stride,
                     q_head_stride,k_head_stride,v_head_stride,
                     o_row_stride,o_head_stride,
                     q_batch_stride,k_batch_stride,v_batch_stride,o_batch_stride,
                     varlen_padded_input,
                     attn_mask,
                     flashmask_downstart_ptr,
                     flashmask_upend_ptr,
                     flashmask_downend_ptr,
                     flashmask_upstart_ptr,
                     flashmask_maxmin_ptr,
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    // Set the pointers and strides.
    params.do_ptr = dout;
    params.do_row_stride = do_row_stride;
    params.do_head_stride = do_head_stride;
    params.dq_ptr = dq;
    params.dk_ptr = dk;
    params.dv_ptr = dv;
    params.dq_row_stride = dq_row_stride;
    params.dk_row_stride = dk_row_stride;
    params.dv_row_stride = dv_row_stride;
    params.dq_head_stride = dq_head_stride;
    params.dk_head_stride = dk_head_stride;
    params.dv_head_stride = dv_head_stride;

    if (cu_seqlens_q_d == nullptr || varlen_padded_input) {
        params.do_batch_stride = do_batch_stride;
        params.dq_batch_stride = dq_batch_stride;
        params.dk_batch_stride = dk_batch_stride;
        params.dv_batch_stride = dv_batch_stride;
    }
    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.num_splits = num_splits;
}

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);

    params.is_causal = is_causal;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     is_causal);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // FP16_SWITCH(!params.is_bf16, [&] {
        // HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(params.attn_mask_ptr != nullptr, Is_attn_mask, [&] {
                    // BOOL_SWITCH(params.flashmask_downstart_ptr != nullptr, Is_sparse_attn_mask, [&] {
                        run_mha_fwd_<cutlass::bfloat16_t, 128, Is_causal, false, true>(params, stream);
                    // });
                });
            });
        // });
    // });
}


bool flash_attn_fwd(const void * const q,
                    const void * const k,
                    const void * const v,
                    void * const rng_state,
                    void * const out,
                    void * const softmax_ptr,
                    void * const softmax_lse_ptr,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool return_softmax,
                    const bool is_bf16,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const int64_t * const flashmask_dims,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int o_batch_stride) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : flashmask_dims ? flashmask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_FWD_EXECTUABLE(seqlen_q, seqlen_k)

    Flash_fwd_params params;
    set_params_fprop_strided(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     return_softmax ? softmax_ptr : nullptr,
                     softmax_lse_ptr,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     q_row_stride,
                     k_row_stride,
                     v_row_stride,
                     q_head_stride,
                     k_head_stride,
                     v_head_stride,
                     o_row_stride,
                     o_head_stride,
                     q_batch_stride,
                     k_batch_stride,
                     v_batch_stride,
                     o_batch_stride,
                     false/*varlen_padded_input=*/,
                     const_cast<void *>(attn_mask),
                     const_cast<void *>(flashmask_downstart_ptr),
                     const_cast<void *>(flashmask_upend_ptr),
                     const_cast<void*>(flashmask_downend_ptr),
                     const_cast<void*>(flashmask_upstart_ptr),
                     const_cast<void *>(flashmask_maxmin_ptr),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    params.rng_state = static_cast<uint64_t*>(rng_state);

    if (is_dropout) {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        params.philox_args = at::PhiloxCudaState(seed, offset);
    }

    run_mha_fwd(params, stream);
    
    return true;

    FLASHATTNLIB_END_FUNC
}

std::vector<at::Tensor>
flashmask_fwd(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        const c10::optional<at::Tensor> &startend_row_indices,
        const c10::optional<at::Tensor> &attn_mask,
        const c10::optional<at::Tensor> &fixed_seed_offset,
        const float dropout,
        bool causal,
        bool return_softmax,
        bool is_test) {
    // auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    // bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    // if (q_dtype == torch::kBFloat16) {
    //     TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    // }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    TORCH_CHECK(q.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "Input tensor must be on CUDA device");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    at::Tensor out;
    out = torch::empty_like(q);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        // TORCH_CHECK(dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    const float softmax_scale = 1.0f / std::sqrt(head_size);
    const float softmax_unscale = std::sqrt(head_size);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

    at::Tensor flashmask_maxmin, downstart_row_indices, upend_row_indices,
                downend_row_indices, upstart_row_indices;
    void *downstart_row_indices_data = nullptr, *upend_row_indices_data = nullptr,
       *downend_row_indices_data = nullptr, *upstart_row_indices_data = nullptr;
    
    bool is_flashmask = startend_row_indices.has_value() && startend_row_indices.value().defined();
    std::vector<int64_t> mask_dim_4d;
    if (is_flashmask) {
        const at::Tensor& startend_row_indices_tensor = startend_row_indices.value();
        auto flashmask_maxmin_shape = startend_row_indices_tensor.sizes().vec();
        flashmask_maxmin_shape[2] = (flashmask_maxmin_shape[2] + 31) / 32 * 8;
        flashmask_maxmin = torch::empty(
            flashmask_maxmin_shape,
            torch::dtype(torch::kInt32).device(torch::kCUDA)
        );
        // 注意调试
        downstart_row_indices = startend_row_indices_tensor.slice(3, 0, 1).clone();
        downstart_row_indices_data = downstart_row_indices.data_ptr();

        if (startend_row_indices_tensor.size(3) == 2) {
            if (!causal) {
                upend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
                upend_row_indices_data = upend_row_indices.data_ptr();
            } else {
                downend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
                downend_row_indices_data = downend_row_indices.data_ptr();
            }
        } else if (startend_row_indices_tensor.size(3) == 4) {
        upend_row_indices = startend_row_indices_tensor.slice(3, 3, 4).clone();
        upend_row_indices_data = upend_row_indices.data_ptr();
        downend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
        downend_row_indices_data = downend_row_indices.data_ptr();
        upstart_row_indices = startend_row_indices_tensor.slice(3, 2, 3).clone();
        upstart_row_indices_data = upstart_row_indices.data_ptr();
        }

        TORCH_CHECK(
                startend_row_indices_tensor.dtype() == torch::kInt32,
                "dtype of startend_row_indices must be int32, but received ",
                startend_row_indices_tensor.dtype());
        const auto& origin_dims = startend_row_indices_tensor.sizes();  // 获取形状（返回c10::IntArrayRef）
        int64_t rank = origin_dims.size();
        int64_t first_dim = 1;
        for (int i = 0; i < rank - 3; ++i) {
            first_dim *= origin_dims[i];
        }
        mask_dim_4d = {
            first_dim,
            origin_dims[rank - 3],
            origin_dims[rank - 2],
            origin_dims[rank - 1]
        };
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    uint64_t seed;
    uint64_t offset;
    if (dropout > 0.0f) {
        if (fixed_seed_offset.has_value()) {
            const at::Tensor& seed_tensor = fixed_seed_offset.value();
            if (seed_tensor.defined()) {
                const int64_t* fixed_seed_offset_data =
                    seed_tensor.data_ptr<int64_t>();
                seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
                offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
            }
        } else {
            // 待完善
            seed = 0;
            offset = 0;
        }
    } else {
        seed = 0;
        offset = 0;
    }

    std::vector<int64_t> attn_mask_dim;
    bool is_attn_mask = attn_mask.has_value() && attn_mask.value().defined();
    const at::Tensor* attn_mask_tensor_ptr;
    if (is_attn_mask) {
        attn_mask_tensor_ptr = &attn_mask.value();
        const auto& origin_dims = attn_mask_tensor_ptr->sizes();
        int64_t rank = origin_dims.size();
        TORCH_CHECK(
            rank >= 4,
            "The number of dimensions of attn_mask is expected to be greater or equal to 4, ",
            "but received ", rank, ". The shape of attn_mask is [",
            c10::Join(", ", origin_dims), "]");
        int64_t first_dim = 1;
        for (int i = 0; i < rank - 3; ++i) {
            first_dim *= origin_dims[i];
        }
        attn_mask_dim = {
            first_dim,
            origin_dims[rank - 3],  // 倒数第三维
            origin_dims[rank - 2],  // 倒数第二维
            origin_dims[rank - 1]   // 最后一维
        };
    }

    flash_attn_fwd(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        reinterpret_cast<uint64_t*>(rng_state.data_ptr()),
        out.data_ptr(),
        return_softmax ? p.data_ptr() : nullptr,
        softmax_lse.data_ptr(),
        batch_size,
        seqlen_q, // max_seqlen_q
        seqlen_k, // max_seqlen_k
        seqlen_q_rounded, // seqlen_q_rounded
        seqlen_k_rounded, // seqlen_k_rounded
        num_heads,
        num_heads_k,
        head_size,
        head_size_rounded,
        is_test ? 0.0f : dropout,
        softmax_scale,
        softmax_unscale,
        causal,
        return_softmax,
        q_dtype == torch::kBFloat16,
        stream,
        seed, // seed
        offset, // offset
        is_attn_mask ? attn_mask_tensor_ptr->data_ptr() : nullptr, // attn_mask_tensor
        is_attn_mask ? attn_mask_dim.data() : nullptr, // mask_dims
        is_flashmask ? downstart_row_indices_data : nullptr,
        is_flashmask ? mask_dim_4d.data() : nullptr,
        is_flashmask ? upend_row_indices_data : nullptr,
        is_flashmask ? downend_row_indices_data : nullptr,
        is_flashmask ? upstart_row_indices_data : nullptr,
        is_flashmask ? flashmask_maxmin.data_ptr() : nullptr,
        q.stride(1), // 注意调试
        k.stride(1),
        v.stride(1),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        out.stride(1),
        out.stride(2),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0)
    );

    return {out, softmax_lse, p, rng_state}; 

}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    // FP16_SWITCH(!params.is_bf16, [&] {
        // HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // BOOL_SWITCH(params.attn_mask_ptr != nullptr, Is_attn_mask, [&] {
                    // BOOL_SWITCH(params.flashmask_downstart_ptr != nullptr, Is_sparse_attn_mask, [&] {
                        run_mha_bwd_<cutlass::bfloat16_t, 128, Is_causal, false, true>(params, stream, configure);
                    // });
                // });
            });
        // });
    // });
}

bool flash_attn_bwd(const void * const dout,
                    const void * const q,
                    const void * const k,
                    const void * const v,
                    const void * const out,
                    const void * const softmax_d,
                    const void * const softmax_lse,
                    void * const rng_state,
                    void * const dq,
                    void * const dk,
                    void * const dv,
                    void * const dq_accum,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool is_bf16,
                    const int num_splits,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const int64_t * const flashmask_dims,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int o_batch_stride,
                    const int dq_row_stride,
                    const int dk_row_stride,
                    const int dv_row_stride,
                    const int dq_head_stride,
                    const int dk_head_stride,
                    const int dv_head_stride,
                    const int do_row_stride,
                    const int do_head_stride,
                    const int dq_batch_stride,
                    const int dk_batch_stride,
                    const int dv_batch_stride,
                    const int do_batch_stride) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : flashmask_dims ? flashmask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_BWD_EXECTUABLE(seqlen_q, seqlen_k)

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    const bool loop = true;

    Flash_bwd_params params;

    set_params_dgrad_strided(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     const_cast<void *>(out),
                     const_cast<void *>(dout),
                     dq,
                     dk,
                     dv,
                     nullptr,
                     nullptr,
                     loop ? dq_accum : nullptr,
                     nullptr,
                     nullptr,
                     const_cast<void *>(softmax_lse),
                     const_cast<void *>(softmax_d),
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     q_row_stride,
                     k_row_stride,
                     v_row_stride,
                     q_head_stride,
                     k_head_stride,
                     v_head_stride,
                     o_row_stride,
                     o_head_stride,
                     q_batch_stride,
                     k_batch_stride,
                     v_batch_stride,
                     o_batch_stride,
                     dq_row_stride,
                     dk_row_stride,
                     dv_row_stride,
                     dq_head_stride,
                     dk_head_stride,
                     dv_head_stride,
                     do_row_stride,
                     do_head_stride,
                     dq_batch_stride,
                     dk_batch_stride,
                     dv_batch_stride,
                     do_batch_stride,
                     false/*varlen_padded_input=*/,
                     num_splits,
                     const_cast<void *>(attn_mask),
                     const_cast<void *>(flashmask_downstart_ptr),
                     const_cast<void *>(flashmask_upend_ptr),
                     const_cast<void*>(flashmask_downend_ptr),
                     const_cast<void*>(flashmask_upstart_ptr),
                     const_cast<void *>(flashmask_maxmin_ptr),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    auto launch = &run_mha_bwd;
    
    if (is_dropout) {
        params.philox_args = at::PhiloxCudaState(seed, offset);
        // seems a wild pointer at fa2: https://github.com/PaddlePaddle/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L690-L691
        params.rng_state = static_cast<uint64_t*>(rng_state);
        uint64_t rng_state_data[2] = {seed, offset};
        cudaMemcpyAsync(params.rng_state, rng_state_data, 2*sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    launch(params, stream, /*configure=*/false);
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

int get_num_split() {
  // 0 for an internal heuristic, which is optimal
  return 1;
}

std::vector<at::Tensor>
flashmask_bwd(const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &softmax_lse,
    const c10::optional<at::Tensor> &startend_row_indices,
    const at::Tensor &dout,
    c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
    c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
    c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
    float dropout,
    bool causal) 
{
    auto q_dtype = q.dtype();
    const auto dims = q.sizes();

    const int64_t batch_size = dims[0];
    const int64_t seqlen_q = dims[1];
    const int64_t num_heads = dims[2];
    const int64_t head_size_og = dout.size(3);
    const int64_t head_size = dims[3];
    const int64_t seqlen_k = k.size(1);
    const int64_t num_heads_k = k.size(2);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    bool is_mha = (num_heads == num_heads_k);

    std::initializer_list<int64_t> dk_dv_shape = {
      batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size};

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        TORCH_CHECK(dq.is_cuda(), "dq must be on CUDA device");
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        TORCH_CHECK(dk.is_cuda(), "dk must be on CUDA device");
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        TORCH_CHECK(dv.is_cuda(), "dv must be on CUDA device");
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(k);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float softmax_scale = 1.0f / std::sqrt(head_size);
    const float softmax_unscale = std::sqrt(head_size);

    // 参数可调试是否选用启发式算法
    int num_splits = get_num_split();

    at::Tensor flashmask_maxmin, downstart_row_indices, upend_row_indices,
                downend_row_indices, upstart_row_indices;
    void *downstart_row_indices_data = nullptr, *upend_row_indices_data = nullptr,
       *downend_row_indices_data = nullptr, *upstart_row_indices_data = nullptr;
    
    bool is_flashmask = startend_row_indices.has_value() && startend_row_indices.value().defined();
    std::vector<int64_t> mask_dim_4d;
    if (is_flashmask) {
        const at::Tensor& startend_row_indices_tensor = startend_row_indices.value();
        auto flashmask_maxmin_shape = startend_row_indices_tensor.sizes().vec();
        flashmask_maxmin_shape[2] = (flashmask_maxmin_shape[2] + 31) / 32 * 8;
        flashmask_maxmin = torch::empty(
            flashmask_maxmin_shape,
            torch::dtype(torch::kInt32).device(torch::kCUDA)
        );
        // 注意调试
        downstart_row_indices = startend_row_indices_tensor.slice(3, 0, 1).clone();
        downstart_row_indices_data = downstart_row_indices.data_ptr();

        if (startend_row_indices_tensor.size(3) == 2) {
            if (!causal) {
                upend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
                upend_row_indices_data = upend_row_indices.data_ptr();
            } else {
                downend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
                downend_row_indices_data = downend_row_indices.data_ptr();
            }
        } else if (startend_row_indices_tensor.size(3) == 4) {
        upend_row_indices = startend_row_indices_tensor.slice(3, 3, 4).clone();
        upend_row_indices_data = upend_row_indices.data_ptr();
        downend_row_indices = startend_row_indices_tensor.slice(3, 1, 2).clone();
        downend_row_indices_data = downend_row_indices.data_ptr();
        upstart_row_indices = startend_row_indices_tensor.slice(3, 2, 3).clone();
        upstart_row_indices_data = upstart_row_indices.data_ptr();
        }

        TORCH_CHECK(
                startend_row_indices_tensor.dtype() == torch::kInt32,
                "dtype of startend_row_indices must be int32, but received ",
                startend_row_indices_tensor.dtype());
        const auto& origin_dims = startend_row_indices_tensor.sizes();  // 获取形状（返回c10::IntArrayRef）
        int64_t rank = origin_dims.size();
        int64_t first_dim = 1;
        for (int i = 0; i < rank - 3; ++i) {
            first_dim *= origin_dims[i];
        }
        mask_dim_4d = {
            first_dim,
            origin_dims[rank - 3],
            origin_dims[rank - 2],
            origin_dims[rank - 1]
        };
    }

    std::vector<int64_t> softmax_lse_dims;
    auto opts = q.options();
    softmax_lse_dims = {batch_size, num_heads, seqlen_q_rounded};
    at::Tensor softmax_d = torch::empty(seqlen_q_rounded, opts.dtype(at::kFloat));

    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));

    at::Tensor dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    flash_attn_bwd(
        dout.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        softmax_d.data_ptr(),
        softmax_lse.data_ptr(),
        rng_state.data_ptr(),
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        dq_accum.data_ptr(),
        batch_size,
        seqlen_q, // max_seqlen_q
        seqlen_k, // max_seqlen_k
        seqlen_q_rounded, // seqlen_q_rounded
        seqlen_k_rounded, // seqlen_k_rounded
        num_heads,
        num_heads_k,
        head_size,
        head_size_rounded,
        dropout,
        softmax_scale,
        softmax_unscale,
        causal,
        q_dtype == torch::kBFloat16,
        num_splits,
        stream,
        0, // seed
        0, // offset
        nullptr, //  attn_mask_tensor
        nullptr, // mask_dims
        is_flashmask ? downstart_row_indices_data : nullptr,
        is_flashmask ? mask_dim_4d.data() : nullptr,
        is_flashmask ? upend_row_indices_data : nullptr,
        is_flashmask ? downend_row_indices_data : nullptr,
        is_flashmask ? upstart_row_indices_data : nullptr,
        is_flashmask ? flashmask_maxmin.data_ptr() : nullptr,
        q.stride(1), // 注意调试
        k.stride(1),
        v.stride(1),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        out.stride(1),
        out.stride(2),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out.stride(0),
        dq.stride(1),
        dk.stride(1),
        dv.stride(1),
        dq.stride(2),
        dk.stride(dk.dim() - 2),
        dv.stride(dk.dim() - 2),
        dout.stride(1),
        dout.stride(2),
        dq.stride(0),
        dk.stride(0),
        dv.stride(0),
        dout.stride(0)
    );

    return { dq, dk, dv, softmax_d };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    // m.def("fwd", &mha_fwd, "Forward pass");
    m.def("flashmask_fwd", &flashmask_fwd, "Flashmask forward pass");
    m.def("flashmask_bwd", &flashmask_bwd, "Flashmask forward pass");
    // m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    // m.def("bwd", &mha_bwd, "Backward pass");
    // m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
}

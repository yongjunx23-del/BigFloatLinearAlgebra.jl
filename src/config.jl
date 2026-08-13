# Explicit kernel configuration. BFLA has no process-global mutable tuning
# state; every non-default kernel choice is carried by an immutable
# `KernelConfig` value supplied by the caller.

"""
    KernelConfig(;
        thread_count = 1,
        gemm_block = 0,
        syrk_block = 0,
        cholesky_block = 0,
        trsm_block = 0,
    )

Immutable, deterministic configuration for the Native kernels.

  * `thread_count`: number of worker tasks for parallel kernels (`1` = serial).
  * `gemm_block`, `syrk_block`, `cholesky_block`, `trsm_block`: blocking factor
    for the corresponding blocked kernel. `0` selects the unblocked kernel,
    which is the conservative default.

No field is read from ambient state; callers that want more threads or blocking
must pass an explicit `KernelConfig`.
"""
struct KernelConfig
    thread_count::Int
    gemm_block::Int
    syrk_block::Int
    cholesky_block::Int
    trsm_block::Int

    function KernelConfig(
        thread_count::Int,
        gemm_block::Int,
        syrk_block::Int,
        cholesky_block::Int,
        trsm_block::Int,
    )
        thread_count >= 1 || throw(ArgumentError(
            "thread_count must be at least 1",
        ))
        for (name, value) in (
            (:gemm_block, gemm_block),
            (:syrk_block, syrk_block),
            (:cholesky_block, cholesky_block),
            (:trsm_block, trsm_block),
        )
            value >= 0 || throw(ArgumentError("$name must be non-negative"))
        end
        return new(
            thread_count,
            gemm_block,
            syrk_block,
            cholesky_block,
            trsm_block,
        )
    end
end

function KernelConfig(; kwargs...)
    allowed = (
        :thread_count,
        :gemm_block,
        :syrk_block,
        :cholesky_block,
        :trsm_block,
    )
    for k in keys(kwargs)
        k in allowed || throw(ArgumentError("unknown KernelConfig field: $k"))
    end
    return KernelConfig(
        get(kwargs, :thread_count, 1),
        get(kwargs, :gemm_block, 0),
        get(kwargs, :syrk_block, 0),
        get(kwargs, :cholesky_block, 0),
        get(kwargs, :trsm_block, 0),
    )
end

Base.show(io::IO, cfg::KernelConfig) =
    print(io, "KernelConfig(thread_count=", cfg.thread_count,
          ", gemm_block=", cfg.gemm_block,
          ", syrk_block=", cfg.syrk_block,
          ", cholesky_block=", cfg.cholesky_block,
          ", trsm_block=", cfg.trsm_block, ")")

@inline _worker_count(config::KernelConfig, jobs::Int) =
    max(1, min(config.thread_count, max(jobs, 1)))

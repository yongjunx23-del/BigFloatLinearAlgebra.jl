# Internal MPFR wrappers used only by the Native backend.
#
# Version assumptions: these wrappers call Julia's MPFR C bindings through
# `Base.MPFR.libmpfr` and `Base.MPFR.rounding_raw(BigFloat)`, which are internal
# to Julia and are exercised by the `test/mpfr.jl` version/precision regression
# tests. They must never escape this module, and the return values are always
# the destination `BigFloat` so callers cannot observe raw pointers.
#
# MPFR rounds into the *destination* object's own precision. Because every
# destination here is a `BigFloat` constructed at an explicit target precision,
# these wrappers never inherit Julia's global `setprecision` context.

"""
    _mpfr_div!(output, numerator, denominator)

`output = numerator / denominator`, correctly rounded into `output`'s
precision. MutableArithmetics does not expose in-place BigFloat division, so
this small wrapper is kept local to the Native kernel layer.
"""
@inline function _mpfr_div!(
    output::BigFloat,
    numerator::BigFloat,
    denominator::BigFloat,
)
    ccall(
        (:mpfr_div, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Ref{BigFloat},
            Ref{BigFloat},
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        numerator,
        denominator,
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

"""
    _mpfr_sqrt!(output, input)

`output = sqrt(input)`, correctly rounded into `output`'s precision.
"""
@inline function _mpfr_sqrt!(
    output::BigFloat,
    input::BigFloat,
)
    ccall(
        (:mpfr_sqrt, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Ref{BigFloat},
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        input,
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

"""
    _mpfr_div_2!(output, input)

`output = input / 2`, correctly rounded into `output`'s precision.
"""
@inline function _mpfr_div_2!(
    output::BigFloat,
    input::BigFloat,
)
    ccall(
        (:mpfr_div_2ui, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Ref{BigFloat},
            Culong,
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        input,
        Culong(1),
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

"""
    _mpfr_set_zero!(output)

`output = 0`, in place, preserving `output`'s precision.
"""
@inline function _mpfr_set_zero!(output::BigFloat)
    ccall(
        (:mpfr_set_zero, Base.MPFR.libmpfr),
        Cint,
        (Ref{BigFloat}, Cint),
        output,
        Cint(0),
    )
    return output
end

"""
    _mpfr_set_ui_2exp!(output, significand, exponent)

Set `output = significand * 2^exponent`, rounded directly into `output`'s
precision. This wrapper is used for exact binary scale constants whose
construction must not inherit ambient `setprecision`.
"""
@inline function _mpfr_set_ui_2exp!(
    output::BigFloat,
    significand::Integer,
    exponent::Integer,
)
    significand >= 0 || throw(ArgumentError(
        "_mpfr_set_ui_2exp!: significand must be nonnegative",
    ))
    ccall(
        (:mpfr_set_ui_2exp, Base.MPFR.libmpfr),
        Cint,
        (
            Ref{BigFloat},
            Culong,
            Clong,
            Base.MPFR.MPFRRoundingMode,
        ),
        output,
        Culong(significand),
        Clong(exponent),
        Base.MPFR.rounding_raw(BigFloat),
    )
    return output
end

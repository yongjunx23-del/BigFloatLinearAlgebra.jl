# Symmetric triangle utilities.

"""
    mirror_triangle!(A, triangle) -> A

Copy the authoritative triangle into the other triangle so that `A` becomes
fully symmetric. For `Lower`, entries above the diagonal are overwritten from
below; for `Upper`, entries below the diagonal are overwritten from above.
Each written entry is an independent MPFR object.
"""
function mirror_triangle!(A::AbstractMatrix{BigFloat}, triangle::Triangle)
    _require_square(A, "mirror_triangle!")
    _require_valid_triangle(triangle, "mirror_triangle!")
    rows = axes(A, 1)
    columns = axes(A, 2)
    @inbounds for column in columns
        for row in rows
            if triangle === Lower
                row < column || continue
                A[row, column] = MA.mutable_copy(A[column, row])
            else
                row > column || continue
                A[row, column] = MA.mutable_copy(A[column, row])
            end
        end
    end
    return A
end

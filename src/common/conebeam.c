/*
 * Copyright (C) 2015-2017 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "conebeam.h"

gfloat
get_float_from_array_or_scalar (GValueArray *array, guint index)
{
    /* *array* is either an array of length 1 and the first value is returned no
     * matter what *index* is, or the *array* length must be sufficient to
     * retrieve *index* */
    if (array->n_values == 1) {
        index = 0;
    } else {
        g_assert (array->n_values > index);
    }

    return EXTRACT_FLOAT (array, index);
}

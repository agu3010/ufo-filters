/*
 * Copyright (C) 2017 Karlsruhe Institute of Technology
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

#include <math.h>
#include "ufo-ctgeometry.h"
#include "ufo-scarray.h"

UfoPoint *
ufo_point_new (UfoScarray *x, UfoScarray *y, UfoScarray *z)
{
    UfoPoint *point = g_new0 (UfoPoint, 1);

    point->x = ufo_scarray_copy (x);
    point->y = ufo_scarray_copy (y);
    point->z = ufo_scarray_copy (z);

    return point;
}

UfoPoint *
ufo_point_copy (const UfoPoint *point)
{
    return ufo_point_new (point->x, point->y, point->z);
}

void
ufo_point_free (UfoPoint *point)
{
    ufo_scarray_free (point->x);
    ufo_scarray_free (point->y);
    ufo_scarray_free (point->z);
    g_free (point);
}

gboolean
ufo_point_are_almost_zero (UfoPoint *point)
{
    return ufo_scarray_is_almost_zero (point->x) &&
           ufo_scarray_is_almost_zero (point->y) &&
           ufo_scarray_is_almost_zero (point->z);
}

UfoVector *
ufo_vector_new (UfoPoint *position, UfoPoint *angle)
{
    UfoVector *vector = g_new0 (UfoVector, 1);
    vector->position = ufo_point_copy (position);
    vector->angle = ufo_point_copy (angle);

    return vector;
}

void
ufo_vector_free (UfoVector *vector)
{
    ufo_point_free (vector->position);
    ufo_point_free (vector->angle);
    g_free (vector);
}

/**
 * Create a new computed tomography geometry with parallel beam geometry.
 */
UfoCTGeometry *
ufo_ctgeometry_new (void)
{
    UfoCTGeometry *geometry = g_new0 (UfoCTGeometry, 1);
    UfoPoint *position, *angle;
    UfoScarray *one, *one_inf;
    GValue double_value = G_VALUE_INIT;

    g_value_init (&double_value, G_TYPE_DOUBLE);
    g_value_set_double (&double_value, -INFINITY);
    one = ufo_scarray_new (1, G_TYPE_DOUBLE, NULL);
    one_inf = ufo_scarray_new (1, G_TYPE_DOUBLE, &double_value);
    position = ufo_point_new (one, one, one);
    angle = ufo_point_new (one, one, one);

    geometry->source_position = ufo_point_new (one, one_inf, one);
    geometry->volume_angle = ufo_point_new (one, one, one);
    geometry->axis = ufo_vector_new (position, angle);
    geometry->detector = ufo_vector_new (position, angle);

    ufo_scarray_free (one);
    ufo_scarray_free (one_inf);
    ufo_point_free (position);
    ufo_point_free (angle);
    g_value_unset (&double_value);

    return geometry;
}

void
ufo_ctgeometry_free (UfoCTGeometry *geometry)
{
    ufo_point_free (geometry->source_position);
    ufo_point_free (geometry->volume_angle);
    ufo_vector_free (geometry->axis);
    ufo_vector_free (geometry->detector);
    g_free (geometry);
}

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
#include "ufo-math.h"
#include "ufo-conebeam.h"
#include "ufo-ctgeometry.h"
#include "ufo-scarray.h"

UfoScpoint *
ufo_scpoint_new (UfoScarray *x, UfoScarray *y, UfoScarray *z)
{
    UfoScpoint *point = g_new0 (UfoScpoint, 1);

    point->x = ufo_scarray_copy (x);
    point->y = ufo_scarray_copy (y);
    point->z = ufo_scarray_copy (z);

    return point;
}

UfoScpoint *
ufo_scpoint_copy (const UfoScpoint *point)
{
    return ufo_scpoint_new (point->x, point->y, point->z);
}

void
ufo_scpoint_free (UfoScpoint *point)
{
    ufo_scarray_free (point->x);
    ufo_scarray_free (point->y);
    ufo_scarray_free (point->z);
    g_free (point);
}

gboolean
ufo_scpoint_are_almost_zero (UfoScpoint *point)
{
    return ufo_scarray_is_almost_zero (point->x) &&
           ufo_scarray_is_almost_zero (point->y) &&
           ufo_scarray_is_almost_zero (point->z);
}

UfoScvector *
ufo_scvector_new (UfoScpoint *position, UfoScpoint *angle)
{
    UfoScvector *vector = g_new0 (UfoScvector, 1);
    vector->position = ufo_scpoint_copy (position);
    vector->angle = ufo_scpoint_copy (angle);

    return vector;
}

void
ufo_scvector_free (UfoScvector *vector)
{
    ufo_scpoint_free (vector->position);
    ufo_scpoint_free (vector->angle);
    g_free (vector);
}

/**
 * Create a new computed tomography geometry with parallel beam geometry.
 */
UfoCTGeometry *
ufo_ctgeometry_new (void)
{
    UfoCTGeometry *geometry = g_new0 (UfoCTGeometry, 1);
    UfoScpoint *position, *angle;
    UfoScarray *one, *one_inf;
    GValue double_value = G_VALUE_INIT;

    g_value_init (&double_value, G_TYPE_DOUBLE);
    g_value_set_double (&double_value, -INFINITY);
    one = ufo_scarray_new (1, G_TYPE_DOUBLE, NULL);
    one_inf = ufo_scarray_new (1, G_TYPE_DOUBLE, &double_value);
    position = ufo_scpoint_new (one, one, one);
    angle = ufo_scpoint_new (one, one, one);

    geometry->source_position = ufo_scpoint_new (one, one_inf, one);
    geometry->volume_angle = ufo_scpoint_new (one, one, one);
    geometry->axis = ufo_scvector_new (position, angle);
    geometry->detector = ufo_scvector_new (position, angle);

    ufo_scarray_free (one);
    ufo_scarray_free (one_inf);
    ufo_scpoint_free (position);
    ufo_scpoint_free (angle);
    g_value_unset (&double_value);

    return geometry;
}

void
ufo_ctgeometry_free (UfoCTGeometry *geometry)
{
    ufo_scpoint_free (geometry->source_position);
    ufo_scpoint_free (geometry->volume_angle);
    ufo_scvector_free (geometry->axis);
    ufo_scvector_free (geometry->detector);
    g_free (geometry);
}

static void
project_voxel (UfoPoint *voxel, UfoPoint *axis_angle, UfoPoint *volume_angle, UfoPoint *detector_angle,
               UfoPoint *detector_position, UfoPoint *source_position, UfoPoint *center_position)
{
    gdouble magnification_recip;
    gdouble tmp, detector_offset;
    gboolean parallel_beam = isinf (source_position->y);
    gboolean perpendicular_detector = detector_angle->x == 0. && detector_angle->y == 0. && detector_angle->z == 0.;
    UfoPoint detector_normal = {0., -1., 0};

    if (!isinf (source_position->y)) {
        magnification_recip = -source_position->y / (detector_position->y - source_position->y);
        ufo_point_mul_scalar (voxel, magnification_recip);
    }

    ufo_point_rotate_z (voxel, volume_angle->z);
    ufo_point_rotate_y (voxel, volume_angle->y);
    ufo_point_rotate_x (voxel, volume_angle->x);

    ufo_point_rotate_z (voxel, axis_angle->z);
    ufo_point_rotate_y (voxel, axis_angle->y);
    ufo_point_rotate_x (voxel, axis_angle->x);

    if (perpendicular_detector) {
        if (!parallel_beam) {
            tmp = (detector_position->y - source_position->y) / (voxel->y - source_position->y);
            ufo_point_mul_scalar (voxel, tmp);
            ufo_point_add (voxel, source_position);
        }
    } else {
        ufo_point_rotate_z (&detector_normal, detector_angle->z);
        ufo_point_rotate_y (&detector_normal, detector_angle->y);
        ufo_point_rotate_x (&detector_normal, detector_angle->x);
        detector_offset = -ufo_point_dot_product (detector_position, &detector_normal);
        if (parallel_beam) {
            voxel->y = - (voxel->z * detector_normal.z + voxel->x * detector_normal.x + detector_offset) / detector_normal.y;
        } else {
            tmp = - (detector_offset + ufo_point_dot_product (source_position, &detector_normal));
            ufo_point_subtract (voxel, source_position);
            ufo_point_mul_scalar (voxel, tmp / ufo_point_dot_product (voxel, &detector_normal));
            ufo_point_add (voxel, source_position);
        }
        ufo_point_subtract (voxel, detector_position);
        ufo_point_rotate_x (voxel, -detector_angle->x);
        ufo_point_rotate_y (voxel, -detector_angle->y);
        ufo_point_rotate_z (voxel, -detector_angle->z);
    }
    ufo_point_add (voxel, center_position);
}

/**
 * Compute projection region required by one slice
 */
static gint *
compute_slice_region (UfoCTGeometry *geometry, gdouble x_min, gdouble x_max, gdouble y_min, gdouble y_max,
                      gsize proj_width, gsize proj_height, gdouble z, guint iteration,
                      UfoUniRecoParameter parameter, gdouble param_value)
{
    gint i, *extrema;
    UfoPoint *axis_angle, *volume_angle, *detector_angle, *detector_position, *source_position,
             *center_position, **voxels;
    gdouble *x_coords, *z_coords;

    voxels = g_new (UfoPoint *, 4);
    x_coords = g_new (gdouble, 4);
    z_coords = g_new (gdouble, 4);
    extrema = g_new (gint, 4);

    /* Setup the angles and positions based on geometry */
    axis_angle = ufo_point_new (ufo_scarray_get_double (geometry->axis->angle->x, iteration),
                                ufo_scarray_get_double (geometry->axis->angle->y, iteration),
                                ufo_scarray_get_double (geometry->axis->angle->z, iteration));
    volume_angle = ufo_point_new (ufo_scarray_get_double (geometry->volume_angle->x, iteration),
                                  ufo_scarray_get_double (geometry->volume_angle->y, iteration),
                                  ufo_scarray_get_double (geometry->volume_angle->z, iteration));
    detector_angle = ufo_point_new (ufo_scarray_get_double (geometry->detector->angle->x, iteration),
                                    ufo_scarray_get_double (geometry->detector->angle->y, iteration),
                                    ufo_scarray_get_double (geometry->detector->angle->z, iteration));
    detector_position = ufo_point_new (ufo_scarray_get_double (geometry->detector->position->x, iteration),
                                       ufo_scarray_get_double (geometry->detector->position->y, iteration),
                                       ufo_scarray_get_double (geometry->detector->position->z, iteration));
    source_position = ufo_point_new (ufo_scarray_get_double (geometry->source_position->x, iteration),
                                     ufo_scarray_get_double (geometry->source_position->y, iteration),
                                     ufo_scarray_get_double (geometry->source_position->z, iteration));
    center_position = ufo_point_new (ufo_scarray_get_double (geometry->axis->position->x, iteration),
                                     ufo_scarray_get_double (geometry->axis->position->y, iteration),
                                     ufo_scarray_get_double (geometry->axis->position->z, iteration));
    /* Adjust the parameter for the third axis to the user defined value */
    switch (parameter) {
        case UFO_UNI_RECO_PARAMETER_AXIS_ROTATION_X:
            axis_angle->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_AXIS_ROTATION_Y:
            axis_angle->y = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_AXIS_ROTATION_Z:
            axis_angle->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_VOLUME_ROTATION_X:
            volume_angle->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_VOLUME_ROTATION_Y:
            volume_angle->y = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_VOLUME_ROTATION_Z:
            volume_angle->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_ROTATION_X:
            detector_angle->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_ROTATION_Y:
            detector_angle->y = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_ROTATION_Z:
            detector_angle->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_POSITION_X:
            detector_position->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_POSITION_Y:
            detector_position->y = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_DETECTOR_POSITION_Z:
            detector_position->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_SOURCE_POSITION_X:
            source_position->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_SOURCE_POSITION_Y:
            source_position->y = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_SOURCE_POSITION_Z:
            source_position->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_CENTER_POSITION_X:
            center_position->x = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_CENTER_POSITION_Z:
            center_position->z = param_value;
            break;
        case UFO_UNI_RECO_PARAMETER_Z:
            z = param_value;
        default:
            break;
    }

    /* Slice x and y coordinates */
    voxels[0] = ufo_point_new (x_min, y_min, z);
    voxels[1] = ufo_point_new (x_min, y_max, z);
    voxels[2] = ufo_point_new (x_max, y_min, z);
    voxels[3] = ufo_point_new (x_max, y_max, z);

    for (i = 0; i < 4; i++) {
        project_voxel (voxels[i], axis_angle, volume_angle, detector_angle, detector_position,
                       source_position, center_position);
        x_coords[i] = voxels[i]->x;
        z_coords[i] = voxels[i]->z;
    }

    /* g_message ("Before rounding: %.2f-%.2f, %.2f-%.2f", ufo_array_minimum (x_coords, 4), */
    /*                                       ufo_array_maximum (x_coords, 4), */
    /*                                       ufo_array_minimum (z_coords, 4), */
    /*                                       ufo_array_maximum (z_coords, 4)); */
    extrema[0] = (gint) ufo_clip_value (floor (ufo_array_minimum (x_coords, 4)), 0., (gdouble) proj_width);
    extrema[1] = (gint) ufo_clip_value (ceil  (ufo_array_maximum (x_coords, 4)), 0., (gdouble) proj_width);
    extrema[2] = (gint) ufo_clip_value (floor (ufo_array_minimum (z_coords, 4)), 0., (gdouble) proj_height);
    extrema[3] = (gint) ufo_clip_value (ceil  (ufo_array_maximum (z_coords, 4)), 0., (gdouble) proj_height);
    /* g_message ("Extrema: %d-%d, %d-%d", extrema[0], extrema[1], extrema[2], extrema[3]); */

    ufo_point_free (axis_angle);
    ufo_point_free (volume_angle);
    ufo_point_free (detector_angle);
    ufo_point_free (detector_position);
    ufo_point_free (source_position);
    ufo_point_free (center_position);
    for (i = 0; i < 4; i++) {
        ufo_point_free (voxels[i]);
    }
    g_free (voxels);
    g_free (x_coords);
    g_free (z_coords);

    return extrema;
}

void
ufo_ctgeometry_compute_projection_region (UfoCTGeometry *geometry, gdouble x_min, gdouble x_max,
                                          gdouble y_min, gdouble y_max, gsize proj_width,
                                          gsize proj_height, gdouble z, guint iteration,
                                          UfoUniRecoParameter parameter, UfoScarray *region)
{
    gint i, *extrema_0, *extrema_1;
    extrema_0 = compute_slice_region (geometry, x_min, x_max, y_min, y_max, proj_width, proj_height,
                                      z, iteration, parameter, ufo_scarray_get_double (region, 0));
    extrema_1 = compute_slice_region (geometry, x_min, x_max, y_min, y_max, proj_width, proj_height,
                                      z, iteration, parameter, ufo_scarray_get_double (region, 1));

    /* g_message ("Region: %g %g %g", ufo_scarray_get_double (region, 0), */
    /*                                ufo_scarray_get_double (region, 1), */
    /*                                ufo_scarray_get_double (region, 2)); */
    /* g_message ("Extrema 1: %d-%d, %d-%d", extrema_0[0], extrema_0[1], extrema_0[2], extrema_0[3]); */
    /* g_message ("Extrema 2: %d-%d, %d-%d", extrema_1[0], extrema_1[1], extrema_1[2], extrema_1[3]); */
    for (i = 0; i < 4; i += 2) {
        extrema_0[i] = MIN (extrema_0[i], extrema_1[i]);
        extrema_0[i + 1] = MAX (extrema_0[i + 1], extrema_1[i + 1]);
    }
    g_message ("Final extrema: %d-%d, %d-%d", extrema_0[0], extrema_0[1], extrema_0[2], extrema_0[3]);
    /* g_message ("%u: %lu x %lu, %g-%g, %g-%g, %d", iteration, proj_width, proj_height, x_min, x_max, y_min, y_max, parameter); */
    /* g_message ("%g %g %g", ufo_scarray_get_double (region, 0), ufo_scarray_get_double (region, 1), ufo_scarray_get_double (region, 2)); */
    g_free (extrema_0);
    g_free (extrema_1);
}

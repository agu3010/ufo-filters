/*
 * Copyright (C) 2011-2015 Karlsruhe Institute of Technology
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <glib/gprintf.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <config.h>
#include "common/conebeam.h"
#include "common/ufo-addressing.h"
#include "ufo-general-backproject-task.h"

#define STATIC_ARG_OFFSET 19
#define G_LOG_LEVEL_DOMAIN "gbp"
#define DEFINE_FILL_SINCOS(type)                      \
static void                                           \
fill_sincos_##type (type *array, const gdouble angle) \
{                                                     \
    array[0] = (type) sin (angle);                    \
    array[1] = (type) cos (angle);                    \
}

#define DEFINE_CREATE_REGIONS(type)                                                 \
static void                                                                         \
create_regions_##type (UfoGeneralBackprojectTaskPrivate *priv,                      \
                       const cl_command_queue cmd_queue,                            \
                       const gdouble start,                                         \
                       const gdouble step)                                          \
{                                                                                   \
    guint i, j;                                                                     \
    gsize region_size;                                                              \
    type *region_values;                                                            \
    cl_int cl_error;                                                                \
    gdouble value;                                                                  \
    gboolean is_angular = is_parameter_angular (priv->parameter);                   \
                                                                                    \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "Start, step: %g %g", start, step);            \
                                                                                    \
    region_size = priv->num_slices_per_chunk * 2 * sizeof (type);                   \
    region_values = (type *) g_malloc0 (region_size);                               \
                                                                                    \
    for (i = 0; i < priv->num_chunks; i++) {                                        \
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Chunk %d region:", i);                    \
        for (j = 0; j < priv->num_slices_per_chunk; j++) {                          \
            value = start + i * priv->num_slices_per_chunk * step + j * step;       \
            if (is_angular) {                                                       \
                region_values[2 * j] = (type) sin (value);                          \
                region_values[2 * j + 1] = (type) cos (value);                      \
            } else {                                                                \
                region_values[2 * j] = (type) value;                                \
            }                                                                       \
            g_log ("gbp", G_LOG_LEVEL_DEBUG, "%g,%g",                               \
                   region_values[2 * j], region_values[2 * j + 1]);                 \
        }                                                                           \
        /* Make sure the memory object is associated with the current node,         \
         * hence no CL_MEM_COPY_HOST_PTR. TODO: If the flag is specified, there are \
         * out of resources errors in the process function. Investigate this. */    \
        priv->cl_regions[i] = clCreateBuffer (priv->context,                        \
                                              CL_MEM_READ_ONLY,                     \
                                              region_size,                          \
                                              NULL,                                 \
                                              &cl_error);                           \
        UFO_RESOURCES_CHECK_CLERR (cl_error);                                       \
        UFO_RESOURCES_CHECK_CLERR (clEnqueueWriteBuffer (cmd_queue,                 \
                                                         priv->cl_regions[i],       \
                                                         CL_TRUE,                   \
                                                         0, region_size,            \
                                                         region_values,             \
                                                         0, NULL, NULL));           \
    }                                                                               \
                                                                                    \
    g_free (region_values);                                                         \
}

#define DEFINE_SET_STATIC_ARGS(type)                                                                                     \
static void                                                                                                              \
set_static_args_##type (UfoGeneralBackprojectTaskPrivate *priv,                                                          \
                        UfoRequisition *requisition,                                                                     \
                        const cl_kernel kernel)                                                                          \
{                                                                                                                        \
    type slice_z_position, region_x[2], region_y[2], axis_x[2], axis_y[2], axis_z[2],                                    \
         volume_x[2], volume_y[2], volume_z[2], detector_x[2], detector_y[2], detector_z[2],                             \
         gray_limit[2], center_position[4], source_position[4], detector_position[4], norm_factor;                       \
    guint burst, j, i = 0;                                                                                               \
    const gint real_size[4] = {requisition->dims[0], requisition->dims[1], (gint) priv->num_slices, 0};                  \
    gdouble gray_delta_recip = (gdouble) get_integer_maximum (st_values[priv->store_type].value_nick) /                  \
                               (priv->gray_map_max - priv->gray_map_min);                                                \
    norm_factor = 2 * G_PI / priv->num_projections;                                                                      \
    burst = kernel == priv->kernel ? BURST : priv->num_projections % BURST;                                              \
                                                                                                                         \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (cl_sampler), &priv->sampler));                       \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (cl_int3), real_size));                               \
                                                                                                                         \
    region_x[0] = (type) EXTRACT_INT (priv->region_x, 0);                                                                \
    region_x[1] = (type) EXTRACT_INT (priv->region_x, 2);                                                                \
    if (!region_x[1]) {                                                                                                  \
        region_x[0] = (type) -requisition->dims[0] / 2.0;                                                                \
        region_x[1] = 1.0f;                                                                                              \
    }                                                                                                                    \
    region_y[0] = (type) EXTRACT_INT (priv->region_y, 0);                                                                \
    region_y[1] = (type) EXTRACT_INT (priv->region_y, 2);                                                                \
    if (!region_y[1]) {                                                                                                  \
        region_y[0] = (type) -requisition->dims[1] / 2.0;                                                                \
        region_y[1] = 1.0f;                                                                                              \
    }                                                                                                                    \
    slice_z_position = (type) priv->z;                                                                                   \
    fill_sincos_##type (axis_x, get_double_from_array_or_scalar (priv->axis_angle_x, priv->count));         \
    fill_sincos_##type (axis_y, get_double_from_array_or_scalar (priv->axis_angle_y, priv->count));         \
    fill_sincos_##type (axis_z, get_double_from_array_or_scalar (priv->axis_angle_z, priv->count));         \
    fill_sincos_##type (volume_x, get_double_from_array_or_scalar (priv->volume_angle_x, priv->count));     \
    fill_sincos_##type (volume_y, get_double_from_array_or_scalar (priv->volume_angle_y, priv->count));     \
    fill_sincos_##type (volume_z, get_double_from_array_or_scalar (priv->volume_angle_z, priv->count));     \
    fill_sincos_##type (detector_x, get_double_from_array_or_scalar (priv->detector_angle_x, priv->count)); \
    fill_sincos_##type (detector_y, get_double_from_array_or_scalar (priv->detector_angle_y, priv->count)); \
    fill_sincos_##type (detector_z, get_double_from_array_or_scalar (priv->detector_angle_z, priv->count)); \
    center_position[0] = (type) get_double_from_array_or_scalar (priv->center_x, priv->count);                           \
    center_position[2] = (type) get_double_from_array_or_scalar (priv->center_z, priv->count);                           \
    /* TODO: use only 2D center in the kernel */                                                                         \
    center_position[1] = 0.0f;                                                                                           \
    source_position[0] = (type) get_double_from_array_or_scalar (priv->source_position_x, priv->count);                  \
    source_position[1] = (type) get_double_from_array_or_scalar (priv->source_position_y, priv->count);                  \
    source_position[2] = (type) get_double_from_array_or_scalar (priv->source_position_z, priv->count);                  \
    detector_position[0] = (type) get_double_from_array_or_scalar (priv->detector_position_x, priv->count);              \
    detector_position[1] = (type) get_double_from_array_or_scalar (priv->detector_position_y, priv->count);              \
    detector_position[2] = (type) get_double_from_array_or_scalar (priv->detector_position_z, priv->count);              \
    norm_factor = (type) norm_factor;                                                                                    \
    gray_limit[0] = (type) priv->gray_map_min;                                                                           \
    gray_limit[1] = (type) gray_delta_recip;                                                                             \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), region_x));                                \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), region_y));                                \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type), &slice_z_position));                          \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), axis_x));                                  \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), axis_y));                                  \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), axis_z));                                  \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), volume_x));                                \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), volume_y));                                \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), volume_z));                                \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), detector_x));                              \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), detector_y));                              \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), detector_z));                              \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##3), center_position));                         \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##3), source_position));                         \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##3), detector_position));                       \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type), &norm_factor));                               \
    UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (type##2), gray_limit));                              \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "region_x: %g %g", region_x[0], region_x[1]);                                       \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "region_y: %g %g", region_y[0], region_y[1]);                                       \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "slice_z_position: %g", slice_z_position);                                          \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "axis: %g %g, %g %g, %g %g",                                                        \
           axis_x[0], axis_x[1], axis_y[0], axis_y[1], axis_z[0], axis_z[1]);                                            \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "volume: %g %g, %g %g, %g %g",                                                      \
           volume_x[0], volume_x[1], volume_y[0], volume_y[1], volume_z[0], volume_z[1]);                                \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "detector_x: %g %g, %g %g, %g %g",                                                  \
           detector_x[0], detector_x[1], detector_y[0], detector_y[1], detector_z[0], detector_z[1]);                    \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "center_position: %g %g %g",                                                        \
           center_position[0], center_position[1], center_position[2]);                                                  \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "source_position: %g %g %g",                                                        \
           source_position[0], source_position[1], source_position[2]);                                                  \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "detector_position: %g %g %g",                                                      \
           detector_position[0], detector_position[1], detector_position[2]);                                            \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "norm_factor: %g", norm_factor);                                                    \
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "gray_limit: %g %g", gray_limit[0], gray_limit[1]);                                 \
                                                                                                                         \
    for (j = 0; j < burst; j++) {                                                                                        \
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, i++, sizeof (cl_mem), &priv->projections[j]));                \
    }                                                                                                                    \
}

/*{{{ Enumerations */
typedef enum {
    PARAMETER_AXIS_ROTATION_X,
    PARAMETER_AXIS_ROTATION_Y,
    PARAMETER_AXIS_ROTATION_Z,
    PARAMETER_VOLUME_ROTATION_X,
    PARAMETER_VOLUME_ROTATION_Y,
    PARAMETER_VOLUME_ROTATION_Z,
    PARAMETER_DETECTOR_POSITION_X,
    PARAMETER_DETECTOR_POSITION_Y,
    PARAMETER_DETECTOR_POSITION_Z,
    PARAMETER_DETECTOR_ROTATION_X,
    PARAMETER_DETECTOR_ROTATION_Y,
    PARAMETER_DETECTOR_ROTATION_Z,
    PARAMETER_SOURCE_POSITION_X,
    PARAMETER_SOURCE_POSITION_Y,
    PARAMETER_SOURCE_POSITION_Z,
    PARAMETER_CENTER_X,
    PARAMETER_CENTER_Z,
    PARAMETER_Z
} Parameter;

typedef enum {
    FT_HALF,
    FT_FLOAT,
    FT_DOUBLE
} FloatType;

typedef enum {
    CT_FLOAT,
    CT_DOUBLE
} ComputeType;

typedef enum {
    ST_HALF,
    ST_FLOAT,
    ST_DOUBLE,
    ST_UCHAR,
    ST_USHORT,
    ST_UINT
} StoreType;

static const GEnumValue parameter_values[] = {
    { PARAMETER_AXIS_ROTATION_X,     "AXIS_ROTATION_X",     "axis_x" },
    { PARAMETER_AXIS_ROTATION_Y,     "AXIS_ROTATION_Y",     "axis_y" },
    { PARAMETER_AXIS_ROTATION_Z,     "AXIS_ROTATION_Z",     "axis_z" },
    { PARAMETER_VOLUME_ROTATION_X,   "VOLUME_ROTATION_X",   "volume_x" },
    { PARAMETER_VOLUME_ROTATION_Y,   "VOLUME_ROTATION_Y",   "volume_y" },
    { PARAMETER_VOLUME_ROTATION_Z,   "VOLUME_ROTATION_Z",   "volume_z" },
    { PARAMETER_DETECTOR_POSITION_X, "DETECTOR_POSITION_X", "detector_x" },
    { PARAMETER_DETECTOR_POSITION_Y, "DETECTOR_POSITION_Y", "detector_y" },
    { PARAMETER_DETECTOR_POSITION_Z, "DETECTOR_POSITION_Z", "detector_z" },
    { PARAMETER_DETECTOR_ROTATION_X, "DETECTOR_ROTATION_X", "detector_position_x" },
    { PARAMETER_DETECTOR_ROTATION_Y, "DETECTOR_ROTATION_Y", "detector_position_y" },
    { PARAMETER_DETECTOR_ROTATION_Z, "DETECTOR_ROTATION_Z", "detector_position_z" },
    { PARAMETER_SOURCE_POSITION_X,   "SOURCE_POSITION_X",   "source_position_x" },
    { PARAMETER_SOURCE_POSITION_Y,   "SOURCE_POSITION_Y",   "source_position_y" },
    { PARAMETER_SOURCE_POSITION_Z,   "SOURCE_POSITION_Z",   "source_position_z" },
    { PARAMETER_CENTER_X,            "CENTER_X",            "center_x" },
    { PARAMETER_CENTER_Z,            "CENTER_Z",            "center_z" },
    { PARAMETER_Z,                   "PARAMETER_Z",         "z" },
    { 0, NULL, NULL}
};

static GEnumValue compute_type_values[] = {
    {CT_FLOAT,  "FT_FLOAT",  "float"},
    {CT_DOUBLE, "FT_DOUBLE", "double"},
    { 0, NULL, NULL}
};

static GEnumValue ft_values[] = {
    {FT_HALF,   "FT_HALF",   "half"},
    {FT_FLOAT,  "FT_FLOAT",  "float"},
    {FT_DOUBLE, "FT_DOUBLE", "double"},
    { 0, NULL, NULL}
};

static GEnumValue st_values[] = {
    {ST_HALF,   "ST_HALF",   "half"},
    {ST_FLOAT,  "ST_FLOAT",  "float"},
    {ST_DOUBLE, "ST_DOUBLE", "double"},
    {ST_UCHAR,  "ST_UCHAR",  "uchar"},
    {ST_USHORT, "ST_USHORT", "ushort"},
    {ST_UINT,   "ST_UINT",   "uint"},
    { 0, NULL, NULL}
};
/*}}}*/

struct _UfoGeneralBackprojectTaskPrivate {
    /* Properties */
    gdouble z;
    GValueArray *region, *region_x, *region_y;
    GValueArray *center_x, *center_z;
    GValueArray *source_position_x, *source_position_y, *source_position_z;
    GValueArray *detector_position_x, *detector_position_y, *detector_position_z;
    GValueArray *detector_angle_x, *detector_angle_y, *detector_angle_z;
    GValueArray *axis_angle_x, *axis_angle_y, *axis_angle_z;
    GValueArray *volume_angle_x, *volume_angle_y, *volume_angle_z;
    ComputeType compute_type, result_type;
    StoreType store_type;
    Parameter parameter;
    gdouble gray_map_min, gray_map_max;
    /* Private */
    guint count, generated;
    cl_mem projections[BURST];
    cl_mem *chunks;
    cl_mem *cl_regions;
    guint num_slices, num_slices_per_chunk, num_chunks;
    gfloat sines[BURST], cosines[BURST];
    guint num_projections;
    gdouble overall_angle;
    AddressingMode addressing_mode;
    /* OpenCL */
    cl_context context;
    cl_kernel kernel, rest_kernel;
    cl_sampler sampler;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoGeneralBackprojectTask, ufo_general_backproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_GENERAL_BACKPROJECT_TASK, UfoGeneralBackprojectTaskPrivate))

enum {
    PROP_0,
    PROP_PARAMETER,
    PROP_Z,
    PROP_REGION,
    PROP_REGION_X,
    PROP_REGION_Y,
    PROP_CENTER_X,
    PROP_CENTER_Z,
    PROP_SOURCE_POSITION_X,
    PROP_SOURCE_POSITION_Y,
    PROP_SOURCE_POSITION_Z,
    PROP_DETECTOR_POSITION_X,
    PROP_DETECTOR_POSITION_Y,
    PROP_DETECTOR_POSITION_Z,
    PROP_DETECTOR_ANGLE_X,
    PROP_DETECTOR_ANGLE_Y,
    PROP_DETECTOR_ANGLE_Z,
    PROP_AXIS_ANGLE_X,
    PROP_AXIS_ANGLE_Y,
    PROP_AXIS_ANGLE_Z,
    PROP_VOLUME_ANGLE_X,
    PROP_VOLUME_ANGLE_Y,
    PROP_VOLUME_ANGLE_Z,
    PROP_NUM_PROJECTIONS,
    PROP_COMPUTE_TYPE,
    PROP_RESULT_TYPE,
    PROP_STORE_TYPE,
    PROP_OVERALL_ANGLE,
    PROP_ADDRESSING_MODE,
    PROP_GRAY_MAP_MIN,
    PROP_GRAY_MAP_MAX,
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

/*{{{ General helper functions*/
DEFINE_FILL_SINCOS (cl_float)
DEFINE_FILL_SINCOS (cl_double)

static gboolean
are_almost_equal (gdouble a, gdouble b)
{
    return (fabs (a - b) < 1e-7);
}

static gsize
get_type_size (StoreType type)
{
    gsize size;

    switch (type) {
        case ST_HALF:
            size = sizeof (cl_half);
            break;
        case ST_FLOAT:
            size = sizeof (cl_float);
            break;
        case ST_DOUBLE:
            size = sizeof (cl_double);
            break;
        case ST_UCHAR:
            size = sizeof (cl_uchar);
            break;
        case ST_USHORT:
            size = sizeof (cl_ushort);
            break;
        case ST_UINT:
            size = sizeof (cl_uint);
            break;
        default:
            g_warning ("Uknown store type");
            size = 0;
            break;
    }

    return size;
}

static gulong
get_integer_maximum (const gchar *type_name)
{
    gint is_uchar = !g_strcmp0 (type_name, "uchar");
    gint is_ushort = !g_strcmp0 (type_name, "ushort");
    gint is_uint = !g_strcmp0 (type_name, "uint");
    gulong maxval = 0;

    if (is_uchar) {
        maxval = 0xFF;
    } else if (is_ushort) {
        maxval = 0xFFFF;
    } else if (is_uint) {
        maxval = 0xFFFFFFFF;
    }

    return maxval;
}

static gboolean
is_parameter_angular (Parameter parameter)
{
    return (parameter == PARAMETER_AXIS_ROTATION_X || parameter == PARAMETER_AXIS_ROTATION_Y ||
            parameter == PARAMETER_AXIS_ROTATION_Z || parameter == PARAMETER_VOLUME_ROTATION_X ||
            parameter == PARAMETER_VOLUME_ROTATION_Y || parameter == PARAMETER_VOLUME_ROTATION_Z ||
            parameter == PARAMETER_DETECTOR_ROTATION_X || parameter == PARAMETER_DETECTOR_ROTATION_Y ||
            parameter == PARAMETER_DETECTOR_ROTATION_Z);
}
/*}}}*/

/*{{{ String Helper functions*/
static gint
find_number_of_occurences (const gchar *haystack, const gchar *needle)
{
    gint num_occurences = 0;
    const gchar *current = haystack;
    guint needle_size = strlen (needle);

    while ((current = strstr (current, needle)) != NULL) {
        current += needle_size;
        num_occurences++;
    }

    return num_occurences;
}

static gchar *
replace_substring (const gchar *haystack, const gchar *needle, const gchar *replacement)
{
    const gchar *current = haystack, *previous = haystack;
    gchar num_occurences = find_number_of_occurences (haystack, needle);
    gint needle_size = strlen (needle);
    gint replacement_size = strlen (replacement);
    gchar *result = g_strnfill (strlen (haystack) + num_occurences *
                                MAX(replacement_size - needle_size, 0), 0);
    gchar *current_result = result;

    while ((current = strstr (previous, needle)) != NULL) {
        /* Copy original from behind the last occurence of needle to the beginning of the next one */
        current_result = g_stpcpy (current_result, g_strndup (previous, (gint) (current - previous)));
        /* Append the replacement substring */
        current_result = g_stpcpy (current_result, replacement);
        /* Continue from behind the needle */
        previous = current + needle_size;
    }
    /* Copy the last chunk of code which doesn't have needle in it */
    g_stpcpy (current_result, previous);

    return result;
}
/*}}}*/

/*{{{ GValueArray helper functions*/
static void
set_region (GValueArray *src, GValueArray **dst)
{
    if (EXTRACT_INT (src, 0) > EXTRACT_INT (src, 1)) {
        g_warning ("Invalid region [\"from\", \"to\", \"step\"]: [%d, %d, %d], "\
                   "\"from\" has to be less than or equal to \"to\"",
                   EXTRACT_INT (src, 0), EXTRACT_INT (src, 1), EXTRACT_INT (src, 2));
    }
    else {
        g_value_array_free (*dst);
        *dst = g_value_array_copy (src);
    }
}

GValueArray *
populate (int num, GType type, gpointer values)
{
    int i;
    GValueArray *array = g_value_array_new (num);
    GValue value = G_VALUE_INIT;
    g_value_init (&value, type);

    for (i = 0; i < num; i++) {
        if (type == G_TYPE_FLOAT) {
            g_value_set_float (&value, *((gfloat *) values + i));
        } else if (type == G_TYPE_DOUBLE) {
            g_value_set_double (&value, *((gdouble *) values + i));
        } else if (type == G_TYPE_INT) {
            g_value_set_int (&value, *((gint *) values + i));
        } else {
            g_warning ("Unknown type '%s'", g_type_name (type));
            g_value_array_free (array);
            return NULL;
        }
        g_value_array_insert (array, i, &value);
    }

    return array;
}
/*}}}*/

/*{{{ Kernel creation*/
/**
 * make_args:
 * @burst: (in): number of processed projections in the kernel
 * @fmt: (in): format string which will be transformed to the projection index
 *
 * Make kernel arguments.
 */
static gchar *
make_args (gint burst, const gchar *fmt)
{
    gint i;
    gulong size, written;
    gchar *one, *str, *ptr;

    size = strlen (fmt) + 1;
    one = g_strnfill (size, 0);
    str = g_strnfill (burst * size, 0);
    ptr = str;

    for (i = 0; i < burst; i++) {
        written = g_snprintf (one, size, fmt, i);
        if (written > size) {
            g_free (one);
            g_free (str);
            return NULL;
        }
        ptr = g_stpcpy (ptr, one);
    }
    g_free (one);

    return str;
}

/**
 * make_type_conversion:
 * @compute_type: (in): data type for internal computations
 * @store_type: (in): output volume data type
 *
 * Make conversions necessary for computation and output data types.
 */
static gchar *
make_type_conversion (const gchar *compute_type, const gchar *store_type)
{
    gulong size = 128;
    gulong written;
    gchar *code = g_strnfill (size, 0);
    gulong maxval = get_integer_maximum (store_type);

    if (maxval) {
        written = g_snprintf (code, size,
                              "(%s) clamp ((%s)(gray_limit.y * (norm_factor * result - gray_limit.x)), (%s) 0.0, (%s) %lu.0)",
                              store_type, compute_type, compute_type, compute_type, maxval);
    } else {
        written = g_snprintf (code, size, "(%s) (norm_factor * result)", store_type);
    }

    if (written > size) {
        g_free (code);
        return NULL;
    }

    return code;
}

/**
 * make_parameter_assignment:
 * @parameter: (in): parameter which represents the third reconstruction axis
 *
 * Make parameter assignment.
 */
static gchar *
make_parameter_assignment (const gchar *parameter)
{
    gchar **entries;
    gchar *code = NULL;

    if (!g_strcmp0 (parameter, "z")) {
        code = g_strdup ("voxel_0.z = region[idz].x;");
    } else if (g_str_has_prefix (parameter, "center") ||
               g_str_has_prefix (parameter, "detector_position") ||
               g_str_has_prefix (parameter, "source_position")) {
        entries = g_strsplit (parameter, "_", 3);
        if (!g_strcmp0 (entries[0], parameter)) {
            /* Not found */
            code = NULL;
        } else {
            if (g_strv_length (entries) == 2) {
                code = g_strconcat (entries[0], ".", entries[1], " = region[idz].x;", NULL);
            } else {
                code = g_strconcat (entries[0], "_", entries[1], ".", entries[2], " = region[idz].x;", NULL);
            }
        }
        g_strfreev (entries);
    } else if (g_str_has_prefix (parameter, "axis") ||
               g_str_has_prefix (parameter, "volume") ||
               g_str_has_prefix (parameter, "detector")) {
        code = g_strconcat (parameter, " = region[idz];", NULL);
    }

    return code;
}

/**
 * make_volume_transformation:
 * @values: (in): sine and cosine angle values
 * @point: 3D point which will be rotated
 *
 * Inplace point rotation about the three coordinate axes.
 */
static gchar *
make_volume_transformation (const gchar *values, const gchar *point)
{
    gulong size = 512;
    gulong written;
    gchar *code = g_strnfill (size, 0);

    written = g_snprintf (code, size,
                          "\t%s = rotate_z (%s_z, %s);"
                          "\n\t%s = rotate_y (%s_y, %s);"
                          "\n\t%s = rotate_x (%s_x, %s);\n",
                          point, values, point,
                          point, values, point,
                          point, values, point);

    if (written > size) {
        g_free (code);
        return NULL;
    }

    return code;
}


/**
 * make_static_transformations:
 * @with_volume: (in): rotate reconstructed volume
 * @perpendicular_detector: (in): is the detector perpendicular to the beam
 * @parallel_beam: (in): is the beam parallel
 *
 * Make static transformations independent from the tomographic rotation angle.
 */
static gchar *
make_static_transformations (gboolean with_volume, gboolean perpendicular_detector, gboolean parallel_beam)
{
    gchar *code = g_strnfill (1024, 0);
    gchar *current = code;
    gchar *detector_transformation, *volume_transformation;

    if (!parallel_beam) {
        current = g_stpcpy (current, "// Magnification\n\tvoxel_0 *= -native_divide(source_position.y, "
                            "(detector_position.y - source_position.y));\n");
    }
    if (!perpendicular_detector) {
        if ((detector_transformation = make_volume_transformation ("detector", "detector_normal")) == NULL) {
            g_free (code);
            return NULL;
        }
        current = g_stpcpy (current, detector_transformation);
        current = g_stpcpy (current, "\n\tdetector_offset = -dot (detector_position, detector_normal);\n");
        g_free (detector_transformation);
    } else if (!parallel_beam) {
        current = g_stpcpy (current, "\n\tproject_tmp = detector_offset - source_position.y;\n");
    }
    if (with_volume) {
        if ((volume_transformation = make_volume_transformation ("volume", "voxel_0")) == NULL) {
            g_free (code);
            return NULL;
        }
        current = g_stpcpy (current, volume_transformation);
        g_free (volume_transformation);
    }
    if (!(perpendicular_detector || parallel_beam)) {
        current = g_stpcpy (current,
                            "\n\ttmp_transformation = "
                            "- (detector_offset + dot (source_position, detector_normal));\n");
    }

    return code;
}

/**
 * make_projection_computation:
 * @perpendicular_detector: (in): is the detector perpendicular to the beam
 * @parallel_beam: (in): is the beam parallel
 *
 * Make voxel projection calculation with the least possible operations based on
 * geometry settings.
 */
static const gchar *
make_projection_computation (gboolean perpendicular_detector, gboolean parallel_beam)
{
    const gchar *code;

    if (perpendicular_detector) {
        if (parallel_beam) {
            code = "\t// Perpendicular detector in combination with parallel beam geometry, i.e.\n"
                   "\t// voxel.xz is directly the detector coordinate, no transformation necessary\n";
        } else {
            code = "\tvoxel = mad (native_divide (project_tmp, (voxel.y - source_position.y)), voxel, source_position);\n";
        }
    } else {
        if (parallel_beam) {
            code = "\tvoxel.y = -native_divide (mad (voxel.z, detector_normal.z, "
                   "mad (voxel.x, detector_normal.x, detector_offset)), detector_normal.y);\n";
        } else {
            code = "\tvoxel -= source_position;\n"
                   "\tvoxel = mad (native_divide (tmp_transformation, dot (voxel, detector_normal)), voxel, source_position);\n";
        }
    }

    return code;
}

/**
 * make_transformations:
 * @burst (in): number of projections processed by the kernel
 * @with_axis: (in): do computations related with rotation axis
 * @perpendicular_detector: (in): is the detector perpendicular to the the mean
 * beam direction
 * @parallel_beam: (in): is the beam parallel
 * @compute_type: (in): data type for internal computations
 *
 * Make voxel projection calculation with the least possible operations based on
 * geometry settings.
 */
static gchar *
make_transformations (gint burst, gboolean with_axis, gboolean perpendicular_detector,
                      gboolean parallel_beam, const gchar *compute_type)
{
    gint i;
    gulong written = 0;
    gchar *code_fmt, *code, *current, *volume_transformation;
    const guint snippet_size = 8192;
    const guint size = burst * snippet_size;
    const gchar *slice_coefficient =
        "\t// Get the value and weigh it (source_position is negative, so -voxel.y\n"
        "\tcoeff = native_divide (source_position.y, (source_position.y - voxel.y));\n";
    const gchar *detector_transformation =
        "\tvoxel -= detector_position;\n"
        "\tvoxel = rotate_x ((cfloat2)(-detector_x.x, detector_x.y), voxel);\n"
        "\tvoxel = rotate_y ((cfloat2)(-detector_y.x, detector_y.y), voxel);\n"
        "\tvoxel = rotate_z ((cfloat2)(-detector_z.x, detector_z.y), voxel);\n";

    code_fmt = g_strnfill (snippet_size, 0);
    code = g_strnfill (size, 0);

    current = g_stpcpy (code_fmt,
                        "\t/* Tomographic rotation angle %02d */"
                        "\n\tvoxel = rotate_z (tomo_%02d, voxel_0);\n");

    if (with_axis) {
        /* Tilted axis of rotation */
        if ((volume_transformation = make_volume_transformation ("axis", "voxel")) == NULL) {
            g_free (code_fmt);
            g_free (code);
            return NULL;
        }
        current = g_stpcpy (current, volume_transformation);
        current = g_stpcpy (current, "\n");
        g_free (volume_transformation);
    }
    if (!parallel_beam) {
        /* FDK normalization computation */
        current = g_stpcpy (current, slice_coefficient);
    }

    /* Voxel projection on the detector */
    current = g_stpcpy (current,
                        "\t// Compute the voxel projected on the detector plane in the global coordinates\n"
                        "\t// V = S + u * (V - S)\n");
    current = g_stpcpy (current, make_projection_computation (perpendicular_detector, parallel_beam));

    if (!perpendicular_detector) {
        /* Transform global coordinates to detector coordinates */
        current = g_stpcpy (current,
                            "\t// Transform the projected coordinates to the detector coordinates, i.e. rotate the\n"
                            "\t// projected voxel to the detector plane\n");
        current = g_stpcpy (current, detector_transformation);
    }

    /* Computational data type adjustment */
    if (!g_strcmp0 (compute_type, "float")) {
        current = g_stpcpy (current,
                            "\tresult += read_imagef (projection_%02d, sampler, "
                            "voxel.xz + center.xz).x");
    } else {
        current = g_stpcpy (current,
                            "\tresult += read_imagef (projection_%02d, sampler, "
                            "convert_float2(voxel.xz + center.xz)).x");
    }

    /* FDK normalization application */
    if (parallel_beam) {
        current = g_stpcpy (current, ";\n\n");
    } else {
        current = g_stpcpy (current, " * coeff * coeff;\n\n");
    }

    for (i = 0; i < burst; i++) {
        written += g_snprintf (code + written, snippet_size, code_fmt, i, i, i);
        if (written > size) {
            g_free (code_fmt);
            g_free (code);
            return NULL;
        }
    }
    g_free (code_fmt);

    return code;
}

/**
 * make_kernel:
 * @template (in): kernel template string
 * @burst (in): how many projections to process in one kernel invocation
 * @with_axis (in): rotate the rotation axis
 * @with_volume: (in): rotate reconstructed volume
 * @perpendicular_detector: (in): is the detector perpendicular to the beam
 * @parallel_beam: (in): is the beam parallel
 * @compute_type (in): data type for calculations (one of "half", "float", "double")
 * @result_type (in): data type for storing the intermediate result (one of "half", "float", "double")
 * @store_type (in): data type of the output volume (one of "half", "float", "double",
 * "uchar", "ushort", "uint")
 * @parameter: (in): parameter which represents the third reconstruction axis
 * @error: A #GError
 *
 * Make backprojection kernel.
 */
static gchar *
make_kernel (gchar *template, gint burst, gboolean with_axis, gboolean with_volume,
             gboolean perpendicular_detector, gboolean parallel_beam, const gchar *compute_type,
             const gchar *result_type, const gchar *store_type, const gchar *parameter,
             GError **error)
{
    const gchar *double_pragma_def, *double_pragma, *half_pragma_def, *half_pragma,
          *image_args_fmt, *trigonomoerty_args_fmt;
    gchar *image_args, *trigonometry_args, *type_conversion, *parameter_assignment,
          *static_transformations, *transformations, *code_tmp, *code, **parts;

    double_pragma_def = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    half_pragma_def = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n";
    image_args_fmt = "\t\t\t read_only image2d_t projection_%02d,\n";
    trigonomoerty_args_fmt = "\t\t\t const cfloat2 tomo_%02d,\n";
    parts = g_strsplit (template, "%tmpl%", 8);

    if ((image_args = make_args (burst, image_args_fmt)) == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Error making image arguments");
        return NULL;
    }
    if ((trigonometry_args = make_args (burst, trigonomoerty_args_fmt)) == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Error making trigonometric arguments");
        return NULL;
    }
    if ((type_conversion = make_type_conversion (compute_type, store_type)) == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Error making type conversion");
        return NULL;
    }
    parameter_assignment = make_parameter_assignment (parameter);
    if (parameter_assignment == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Wrong parameter name");
        return NULL;
    }

    if ((static_transformations = make_static_transformations(with_volume, perpendicular_detector,
                                                              parallel_beam)) == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Error making static transformations");
        return NULL;
    }
    if ((transformations = make_transformations (burst, with_axis, perpendicular_detector,
                                            parallel_beam, compute_type)) == NULL) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP,
                     "Error making tomographic-angle-based transformations");
        return NULL;
    }
    if (!(g_strcmp0 (compute_type, "double") && g_strcmp0 (result_type, "double"))) {
        double_pragma = double_pragma_def;
    } else {
        double_pragma = "";
    }
    if (!(g_strcmp0 (compute_type, "half") && g_strcmp0 (result_type, "half"))) {
        half_pragma = half_pragma_def;
    } else {
        half_pragma = "";
    }
    code_tmp = g_strconcat (double_pragma, half_pragma, parts[0], image_args,
                                 parts[1], trigonometry_args,
                                 parts[2], parameter_assignment,
                                 parts[3], static_transformations,
                                 parts[4], transformations,
                                 parts[5], type_conversion,
                                 parts[6], type_conversion,
                                 parts[7], NULL);
    code = replace_substring (code_tmp, "cfloat", compute_type);
    g_free (code_tmp);
    code_tmp = replace_substring (code, "rtype", result_type);
    g_free (code);
    code = replace_substring (code_tmp, "stype", store_type);

    g_free (image_args);
    g_free (trigonometry_args);
    g_free (type_conversion);
    g_free (parameter_assignment);
    g_free (static_transformations);
    g_free (transformations);
    g_free (code_tmp);
    g_strfreev (parts);

    return code;
}
/*}}}*/

/*{{{ OpenCL helper functions */
DEFINE_CREATE_REGIONS (cl_float)
DEFINE_CREATE_REGIONS (cl_double)

static void
create_images (UfoGeneralBackprojectTaskPrivate *priv, gsize width, gsize height)
{
    cl_image_format image_fmt;
    cl_int cl_error;
    guint i;

    g_log ("gbp", G_LOG_LEVEL_DEBUG, "Creating images %lu x %lu", width, height);

    for (i = 0; i < BURST; i++) {
        /* TODO: dangerous, don't rely on the ufo-buffer */
        image_fmt.image_channel_order = CL_INTENSITY;
        image_fmt.image_channel_data_type = CL_FLOAT;
        /* TODO: what about the "other" API? */
        priv->projections[i] = clCreateImage2D (priv->context,
                                                    CL_MEM_READ_ONLY,
                                                    &image_fmt,
                                                    width,
                                                    height,
                                                    0,
                                                    NULL,
                                                    &cl_error);
        UFO_RESOURCES_CHECK_CLERR (cl_error);
    }
}

DEFINE_SET_STATIC_ARGS (cl_float)
DEFINE_SET_STATIC_ARGS (cl_double)

static void
copy_to_image (const cl_command_queue cmd_queue,
               UfoBuffer *input,
               cl_mem output,
               guint width,
               guint height)
{
    cl_event event;
    cl_int errcode;
    cl_mem input_array;
    const size_t origin[] = {0, 0, 0};
    const size_t region[] = {width, height, 1};

    input_array = ufo_buffer_get_device_array (input, cmd_queue);
    errcode = clEnqueueCopyBufferToImage (cmd_queue,
                                          input_array,
                                          output,
                                          0, origin, region,
                                          0, NULL, &event);

    UFO_RESOURCES_CHECK_CLERR (errcode);
    UFO_RESOURCES_CHECK_CLERR (clWaitForEvents (1, &event));
    UFO_RESOURCES_CHECK_CLERR (clReleaseEvent (event));
}
/*}}}*/

UfoNode *
ufo_general_backproject_task_new (void)
{
    return UFO_NODE (g_object_new (UFO_TYPE_GENERAL_BACKPROJECT_TASK, NULL));
}

static void
ufo_general_backproject_task_setup (UfoTask *task,
                                    UfoResources *resources,
                                    GError **error)
{
    guint i;
    cl_int cl_error;
    gboolean with_axis, with_volume, parallel_beam, perpendicular_detector;
    gchar *template, *kernel_code;
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);

    /* Check parameter values */
    if (!priv->num_projections) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP,
                     "Number of projections not set");
        return;
    }

    if (priv->gray_map_min >= priv->gray_map_max &&
        (priv->store_type == ST_UCHAR || priv->store_type == ST_USHORT || priv->store_type == ST_UINT)) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP,
                     "Gray mapping minimum must be less then the maximum");
        return;
    }

    /* Initialization */
    /* Assume the most efficient geometry, change if necessary */
    with_axis = FALSE;
    with_volume = FALSE;
    perpendicular_detector = TRUE;
    parallel_beam = TRUE;
    priv->kernel = NULL;
    priv->rest_kernel = NULL;
    priv->chunks = NULL;
    priv->cl_regions = NULL;

    /* Actual parameter setup */
    for (i = 0; i < priv->num_projections; i++) {
        if (!(are_almost_equal (get_double_from_array_or_scalar (priv->axis_angle_x, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->axis_angle_y, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->axis_angle_z, 0), 0))) {
            with_axis = TRUE;
        }
        if (!(are_almost_equal (get_double_from_array_or_scalar (priv->volume_angle_x, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->volume_angle_y, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->volume_angle_z, 0), 0))) {
            with_volume = TRUE;
        }
        if (!(are_almost_equal (get_double_from_array_or_scalar (priv->detector_angle_x, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->detector_angle_y, 0), 0) &&
              are_almost_equal (get_double_from_array_or_scalar (priv->detector_angle_z, 0), 0))) {
            perpendicular_detector = FALSE;
        }
        if (!isinf (get_double_from_array_or_scalar (priv->source_position_y, i))) {
            parallel_beam = FALSE;
        }
    }

    g_log ("gbp", G_LOG_LEVEL_DEBUG, "burst: %d, parameter: %s with axis: %d, with volume: %d, "
           "perpendicular detector: %d, parallel beam: %d, "
           "compute type: %s, result type: %s, store type: %s",
             BURST, parameter_values[priv->parameter].value_nick, with_axis, with_volume,
             perpendicular_detector, parallel_beam,
             compute_type_values[priv->compute_type].value_nick,
             ft_values[priv->result_type].value_nick,
             st_values[priv->store_type].value_nick);

    if (priv->axis_angle_x->n_values == priv->num_projections ||
        priv->axis_angle_y->n_values == priv->num_projections ||
        priv->axis_angle_z->n_values == priv->num_projections ||
        priv->volume_angle_x->n_values == priv->num_projections ||
        priv->volume_angle_y->n_values == priv->num_projections ||
        priv->volume_angle_z->n_values == priv->num_projections ||
        priv->detector_angle_x->n_values == priv->num_projections ||
        priv->detector_angle_y->n_values == priv->num_projections ||
        priv->detector_angle_z->n_values == priv->num_projections ||
        priv->detector_position_x->n_values == priv->num_projections ||
        priv->detector_position_y->n_values == priv->num_projections ||
        priv->detector_position_z->n_values == priv->num_projections ||
        priv->source_position_x->n_values == priv->num_projections ||
        priv->source_position_y->n_values == priv->num_projections ||
        priv->source_position_z->n_values == priv->num_projections ||
        priv->center_x->n_values == priv->num_projections ||
        priv->center_z->n_values == priv->num_projections) {
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Using vectorized parameters kernel");
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP, "Vectorized parameters are not yet implemented");
    } else {
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Using scalar-based parameters kernel");
        /* Create kernel source code based on geometry settings */
        if (!(template = ufo_resources_get_kernel_source (resources, "general_backproject.in", error))) {
            return;
        }

        kernel_code = make_kernel (template, BURST, with_axis, with_volume,
                                   perpendicular_detector, parallel_beam,
                                   compute_type_values[priv->compute_type].value_nick,
                                   ft_values[priv->result_type].value_nick,
                                   st_values[priv->store_type].value_nick,
                                   parameter_values[priv->parameter].value_nick, error);
        priv->kernel = ufo_resources_get_kernel_from_source_with_opts (resources,
                                                                       kernel_code,
                                                                       "backproject",
                                                                       NULL,
                                                                       error);
        g_free (kernel_code);

        if (priv->num_projections % BURST) {
            kernel_code = make_kernel (template, priv->num_projections % BURST,
                                       with_axis, with_volume,
                                       perpendicular_detector, parallel_beam,
                                       compute_type_values[priv->compute_type].value_nick,
                                       ft_values[priv->result_type].value_nick,
                                       st_values[priv->store_type].value_nick,
                                       parameter_values[priv->parameter].value_nick, error);

            /* If num_projections % BURST != 0 we need one more kernel to process the remaining projections */
            priv->rest_kernel = ufo_resources_get_kernel_from_source_with_opts (resources,
                                                                                kernel_code,
                                                                                "backproject",
                                                                                NULL,
                                                                                error);
            /* g_printf ("%s", kernel_code); */
            g_free (kernel_code);
        }
        g_free (template);
    }

    for (i = 0; i < BURST; i++) {
        priv->projections[i] = NULL;
    }

    /* Set OpenCL variables */
    priv->context = ufo_resources_get_context (resources);
    UFO_RESOURCES_CHECK_CLERR (clRetainContext (priv->context));

    if (priv->kernel) {
        UFO_RESOURCES_CHECK_CLERR (clRetainKernel (priv->kernel));
    }
    if (priv->rest_kernel) {
        UFO_RESOURCES_CHECK_CLERR (clRetainKernel (priv->rest_kernel));
    }
    priv->sampler = clCreateSampler (priv->context, (cl_bool) FALSE, priv->addressing_mode, CL_FILTER_LINEAR, &cl_error);
    UFO_RESOURCES_CHECK_CLERR (cl_error);
}

static void
ufo_general_backproject_task_get_requisition (UfoTask *task,
                                 UfoBuffer **inputs,
                                 UfoRequisition *requisition)
{
    UfoGeneralBackprojectTaskPrivate *priv;
    UfoRequisition in_req;
    GValue g_value_int = G_VALUE_INIT;
    g_value_init (&g_value_int, G_TYPE_INT);

    priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);
    g_assert (priv->region->n_values == 3);
    requisition->n_dims = 2;
    ufo_buffer_get_requisition (inputs[0], &in_req);

    if (EXTRACT_INT (priv->region_x, 2) == 0) {
        /* If the slice width is not set, reconstruct full width */
        requisition->dims[0] = in_req.dims[0];
    } else {
        requisition->dims[0] = REGION_SIZE (priv->region_x);
    }
    if (EXTRACT_INT (priv->region_y, 2) == 0) {
        /* If the slice height is not set, reconstruct full height, which is the
         * same as width */
        requisition->dims[1] = in_req.dims[0];
    } else {
        requisition->dims[1] = REGION_SIZE (priv->region_y);
    }

    g_log ("gbp", G_LOG_LEVEL_DEBUG, "requisition (x, y, z): %lu %lu %d", requisition->dims[0], requisition->dims[1], 1);
}

static guint
ufo_general_backproject_task_get_num_inputs (UfoTask *task)
{
    return 1;
}

static guint
ufo_general_backproject_task_get_num_dimensions (UfoTask *task,
                                             guint input)
{
    g_return_val_if_fail (input == 0, 0);

    return 2;
}

static UfoTaskMode
ufo_general_backproject_task_get_mode (UfoTask *task)
{
    return UFO_TASK_MODE_REDUCTOR | UFO_TASK_MODE_GPU;
}

static gboolean
ufo_general_backproject_task_process (UfoTask *task,
                                      UfoBuffer **inputs,
                                      UfoBuffer *output,
                                      UfoRequisition *requisition)
{
    UfoGeneralBackprojectTaskPrivate *priv;
    UfoRequisition in_req;
    UfoGpuNode *node;
    UfoProfiler *profiler;
    guint i, index, ki;
    guint burst;
    gsize slice_size, chunk_size, volume_size, projections_size;
    gdouble region_start, region_stop, region_step;
    cl_int cl_error;
    GValue *max_global_mem_size_gvalue, *max_mem_alloc_size_gvalue;
    cl_ulong max_global_mem_size, max_mem_alloc_size;
    cl_kernel kernel;
    cl_command_queue cmd_queue;
    gdouble rot_angle;
    cl_float f_tomo_angle[2];
    cl_double d_tomo_angle[2];
    cl_int cumulate;
    const gsize local_work_size[3] = {16, 8, 8};
    gsize global_work_size[3];
    typedef void (*CreateRegionFunc) (UfoGeneralBackprojectTaskPrivate *, const cl_command_queue,
                                      const gdouble, const gdouble);
    typedef void (*SetStaticArgsFunc) (UfoGeneralBackprojectTaskPrivate *, UfoRequisition *, const cl_kernel);
    CreateRegionFunc create_regions[2] = {create_regions_cl_float, create_regions_cl_double};
    SetStaticArgsFunc set_static_args[2] = {set_static_args_cl_float, set_static_args_cl_double};

    priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);
    node = UFO_GPU_NODE (ufo_task_node_get_proc_node (UFO_TASK_NODE (task)));
    cmd_queue = ufo_gpu_node_get_cmd_queue (node);
    ufo_buffer_get_requisition (inputs[0], &in_req);
    profiler = ufo_task_node_get_profiler (UFO_TASK_NODE (task));
    if (priv->count >= priv->num_projections / BURST * BURST) {
        kernel = priv->rest_kernel;
        burst = priv->num_projections % BURST;
    } else {
        kernel = priv->kernel;
        burst = BURST;
    }

    index = priv->count % burst;
    if (are_almost_equal (EXTRACT_DOUBLE (priv->region, 2), 0.0f)) {
        /* Conservative approach, reconstruct just one slice */
        region_start = 0.0f;
        region_stop = 1.0f;
        region_step = 1.0f;
    } else {
        region_start = EXTRACT_DOUBLE (priv->region, 0);
        region_stop = EXTRACT_DOUBLE (priv->region, 1);
        region_step = EXTRACT_DOUBLE (priv->region, 2);
    }
    if (!priv->num_slices) {
        priv->num_slices = (gsize) ceil ((region_stop - region_start) / region_step);
    }
    max_global_mem_size_gvalue = ufo_gpu_node_get_info (node, UFO_GPU_NODE_INFO_GLOBAL_MEM_SIZE);
    max_global_mem_size = g_value_get_ulong (max_global_mem_size_gvalue);
    g_value_unset (max_global_mem_size_gvalue);
    projections_size = BURST * in_req.dims[0] * in_req.dims[1] * sizeof (cl_float);
    slice_size = requisition->dims[0] * requisition->dims[1] * get_type_size (priv->store_type);
    volume_size = slice_size * priv->num_slices;
    max_mem_alloc_size_gvalue = ufo_gpu_node_get_info (node, UFO_GPU_NODE_INFO_MAX_MEM_ALLOC_SIZE);
    max_mem_alloc_size = g_value_get_ulong (max_mem_alloc_size_gvalue);
    g_value_unset (max_mem_alloc_size_gvalue);
    priv->num_slices_per_chunk = (guint) floor ((gdouble) MIN (max_mem_alloc_size, volume_size) / ((gdouble) slice_size));
    global_work_size[0] = requisition->dims[0] % local_work_size[0] ?
                          NEXT_DIVISOR (requisition->dims[0], local_work_size[0]) :
                          requisition->dims[0];
    global_work_size[1] = requisition->dims[1] % local_work_size[1] ?
                          NEXT_DIVISOR (requisition->dims[1], local_work_size[1]) :
                          requisition->dims[1];
    global_work_size[2] = priv->num_slices_per_chunk % local_work_size[2] ?
                          NEXT_DIVISOR (priv->num_slices_per_chunk, local_work_size[2]) :
                          priv->num_slices_per_chunk;
    if (!priv->count) {
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Global work size: %lu %lu %lu, local: %lu %lu %lu",
               global_work_size[0], global_work_size[1], global_work_size[2],
               local_work_size[0], local_work_size[1], local_work_size[2]);
    }
    if (projections_size + volume_size > max_global_mem_size) {
        g_warning ("Volume size doesn't fit to memory");
        return FALSE;
    }
    if (!priv->chunks) {
        /* Create subvolumes (because one large volume might be larger than the maximum allocatable memory chunk */
        priv->num_chunks = (priv->num_slices - 1) / priv->num_slices_per_chunk + 1;
        chunk_size = priv->num_slices_per_chunk * slice_size;
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Max alloc size: %lu, max global size: %lu", max_mem_alloc_size, max_global_mem_size);
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Chunk size: %lu, num chunks: %d, num slices per chunk: %u",
               chunk_size, priv->num_chunks, priv->num_slices_per_chunk);
        g_log ("gbp", G_LOG_LEVEL_DEBUG, "Volume size: %lu, num slices: %u", volume_size, priv->num_slices);
        priv->chunks = (cl_mem *) g_malloc0 (priv->num_chunks * sizeof (cl_mem));
        priv->cl_regions = (cl_mem *) g_malloc0 (priv->num_chunks * sizeof (cl_mem));
        for (i = 0; i < priv->num_chunks; i++) {
            g_log ("gbp", G_LOG_LEVEL_DEBUG, "Creating chunk %d with size %lu",
                   i, MIN (volume_size, (i + 1) * chunk_size) - i * chunk_size);
            priv->chunks[i] = clCreateBuffer (priv->context,
                                              CL_MEM_WRITE_ONLY,
                                              MIN (volume_size, (i + 1) * chunk_size) - i * chunk_size,
                                              NULL,
                                              &cl_error);
            UFO_RESOURCES_CHECK_CLERR (cl_error);
        }
        create_images (priv, in_req.dims[0], in_req.dims[1]);
        create_regions[priv->compute_type] (priv, cmd_queue, region_start, region_step);
        set_static_args[priv->compute_type] (priv, requisition, priv->kernel);
        set_static_args[priv->compute_type] (priv, requisition, priv->rest_kernel);
    }

    /* Setup tomographic rotation angle dependent arguments */
    copy_to_image (cmd_queue, inputs[0], priv->projections[index], in_req.dims[0], in_req.dims[1]);
    ki = STATIC_ARG_OFFSET + burst;
    rot_angle = ((gdouble) priv->count) / priv->num_projections * priv->overall_angle;
    if (priv->compute_type == CT_FLOAT) {
        fill_sincos_cl_float (f_tomo_angle, rot_angle);
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, ki + index, sizeof (cl_float2), f_tomo_angle));
    } else {
        fill_sincos_cl_double (d_tomo_angle, rot_angle);
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, ki + index, sizeof (cl_double2), d_tomo_angle));
    }

    if (index + 1 == burst) {
        ki += index + 1;
        cumulate = priv->count > burst;
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, ki++, sizeof (cl_int), &cumulate));
        for (i = 0; i < priv->num_chunks; i++) {
            UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, ki, sizeof (cl_mem), &priv->chunks[i]));
            UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, ki + 1, sizeof (cl_mem), &priv->cl_regions[i]));
            ufo_profiler_call (profiler, cmd_queue, kernel, 3, global_work_size, local_work_size);
        }
    }

    priv->count++;

    return TRUE;
}

static gboolean
ufo_general_backproject_task_generate (UfoTask *task,
                         UfoBuffer *output,
                         UfoRequisition *requisition)
{
    UfoGeneralBackprojectTaskPrivate *priv;
    UfoGpuNode *node;
    cl_command_queue cmd_queue;
    cl_mem out_mem;
    guint chunk_index;
    /* TODO: handle other data types */
    size_t bpp;
    size_t src_row_pitch, src_slice_pitch;
    size_t src_origin[3] = {0, 0, 0};
    size_t dst_origin[3] = {0, 0, 0};
    size_t region[3] = {0, 0, 1};

    priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);
    node = UFO_GPU_NODE (ufo_task_node_get_proc_node (UFO_TASK_NODE (task)));
    cmd_queue = ufo_gpu_node_get_cmd_queue (node);
    out_mem = ufo_buffer_get_device_array (output, cmd_queue);
    chunk_index = priv->generated / priv->num_slices_per_chunk;
    bpp = get_type_size (priv->store_type);

    if (priv->generated >= priv->num_slices) {
        return FALSE;
    }

    src_row_pitch = requisition->dims[0] * bpp;
    src_slice_pitch = src_row_pitch * requisition->dims[1];
    src_origin[2] = priv->generated % priv->num_slices_per_chunk;
    region[0] = src_row_pitch;
    region[1] = requisition->dims[1];
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "Generating slice %u from chunk %u", priv->generated + 1, chunk_index);
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "src_origin: %lu %lu %lu", src_origin[0], src_origin[1], src_origin[2]);
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "region: %lu %lu %lu", region[0], region[1], region[2]);
    g_log ("gbp", G_LOG_LEVEL_DEBUG, "row pitch %lu, slice pitch %lu", src_row_pitch, src_slice_pitch);

    UFO_RESOURCES_CHECK_CLERR (clEnqueueCopyBufferRect (cmd_queue,
                                                        priv->chunks[chunk_index], out_mem,
                                                        src_origin, dst_origin, region,
                                                        src_row_pitch, src_slice_pitch,
                                                        src_row_pitch, 0,
                                                        0, NULL, NULL));

    /* TODO: could we do priv->count--? */
    priv->generated++;

    return TRUE;
}

/*{{{ Setters and getters and properties initialization */
static void
ufo_general_backproject_task_set_property (GObject *object,
                              guint property_id,
                              const GValue *value,
                              GParamSpec *pspec)
{
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (object);
    GValueArray *array;

    switch (property_id) {
        case PROP_PARAMETER:
            priv->parameter = g_value_get_enum (value);
            break;
        case PROP_Z:
            priv->z = g_value_get_double (value);
            break;
        case PROP_REGION:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->region);
            priv->region = g_value_array_copy (array);
            break;
        case PROP_REGION_X:
            array = (GValueArray *) g_value_get_boxed (value);
            set_region (array, &priv->region_x);
            break;
        case PROP_REGION_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            set_region (array, &priv->region_y);
            break;
        case PROP_CENTER_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->center_x);
            priv->center_x = g_value_array_copy (array);
            break;
        case PROP_CENTER_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->center_z);
            priv->center_z = g_value_array_copy (array);
            break;
        case PROP_SOURCE_POSITION_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->source_position_x);
            priv->source_position_x = g_value_array_copy (array);
            break;
        case PROP_SOURCE_POSITION_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->source_position_y);
            priv->source_position_y = g_value_array_copy (array);
            break;
        case PROP_SOURCE_POSITION_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->source_position_z);
            priv->source_position_z = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_POSITION_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_position_x);
            priv->detector_position_x = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_POSITION_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_position_y);
            priv->detector_position_y = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_POSITION_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_position_z);
            priv->detector_position_z = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_ANGLE_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_angle_x);
            priv->detector_angle_x = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_ANGLE_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_angle_y);
            priv->detector_angle_y = g_value_array_copy (array);
            break;
        case PROP_DETECTOR_ANGLE_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->detector_angle_z);
            priv->detector_angle_z = g_value_array_copy (array);
            break;
        case PROP_AXIS_ANGLE_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->axis_angle_x);
            priv->axis_angle_x = g_value_array_copy (array);
            break;
        case PROP_AXIS_ANGLE_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->axis_angle_y);
            priv->axis_angle_y = g_value_array_copy (array);
            break;
        case PROP_AXIS_ANGLE_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->axis_angle_z);
            priv->axis_angle_z = g_value_array_copy (array);
            break;
        case PROP_VOLUME_ANGLE_X:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->volume_angle_x);
            priv->volume_angle_x = g_value_array_copy (array);
            break;
        case PROP_VOLUME_ANGLE_Y:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->volume_angle_y);
            priv->volume_angle_y = g_value_array_copy (array);
            break;
        case PROP_VOLUME_ANGLE_Z:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->volume_angle_z);
            priv->volume_angle_z = g_value_array_copy (array);
            break;
        case PROP_NUM_PROJECTIONS:
            priv->num_projections = g_value_get_uint (value);
            break;
        case PROP_COMPUTE_TYPE:
            priv->compute_type = g_value_get_enum (value);
            break;
        case PROP_RESULT_TYPE:
            priv->result_type = g_value_get_enum (value);
            break;
        case PROP_STORE_TYPE:
            priv->store_type = g_value_get_enum (value);
            break;
        case PROP_OVERALL_ANGLE:
            priv->overall_angle = g_value_get_double (value);
            break;
        case PROP_ADDRESSING_MODE:
            priv->addressing_mode = g_value_get_enum (value);
            break;
        case PROP_GRAY_MAP_MIN:
            priv->gray_map_min = g_value_get_double (value);
            break;
        case PROP_GRAY_MAP_MAX:
            priv->gray_map_max = g_value_get_double (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_general_backproject_task_get_property (GObject *object,
                              guint property_id,
                              GValue *value,
                              GParamSpec *pspec)
{
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_PARAMETER:
            g_value_set_enum (value, priv->parameter);
            break;
        case PROP_Z:
            g_value_set_double (value, priv->z);
            break;
        case PROP_REGION:
            g_value_set_boxed (value, priv->region);
            break;
        case PROP_REGION_X:
            g_value_set_boxed (value, priv->region_x);
            break;
        case PROP_REGION_Y:
            g_value_set_boxed (value, priv->region_y);
            break;
        case PROP_CENTER_X:
            g_value_set_boxed (value, priv->center_x);
            break;
        case PROP_CENTER_Z:
            g_value_set_boxed (value, priv->center_z);
            break;
        case PROP_SOURCE_POSITION_X:
            g_value_set_boxed (value, priv->source_position_x);
            break;
        case PROP_SOURCE_POSITION_Y:
            g_value_set_boxed (value, priv->source_position_y);
            break;
        case PROP_SOURCE_POSITION_Z:
            g_value_set_boxed (value, priv->source_position_z);
            break;
        case PROP_DETECTOR_POSITION_X:
            g_value_set_boxed (value, priv->detector_position_x);
            break;
        case PROP_DETECTOR_POSITION_Y:
            g_value_set_boxed (value, priv->detector_position_y);
            break;
        case PROP_DETECTOR_POSITION_Z:
            g_value_set_boxed (value, priv->detector_position_z);
            break;
        case PROP_DETECTOR_ANGLE_X:
            g_value_set_boxed (value, priv->detector_angle_x);
            break;
        case PROP_DETECTOR_ANGLE_Y:
            g_value_set_boxed (value, priv->detector_angle_y);
            break;
        case PROP_DETECTOR_ANGLE_Z:
            g_value_set_boxed (value, priv->detector_angle_z);
            break;
        case PROP_AXIS_ANGLE_X:
            g_value_set_boxed (value, priv->axis_angle_x);
            break;
        case PROP_AXIS_ANGLE_Y:
            g_value_set_boxed (value, priv->axis_angle_y);
            break;
        case PROP_AXIS_ANGLE_Z:
            g_value_set_boxed (value, priv->axis_angle_z);
            break;
        case PROP_VOLUME_ANGLE_X:
            g_value_set_boxed (value, priv->volume_angle_x);
            break;
        case PROP_VOLUME_ANGLE_Y:
            g_value_set_boxed (value, priv->volume_angle_y);
            break;
        case PROP_VOLUME_ANGLE_Z:
            g_value_set_boxed (value, priv->volume_angle_z);
            break;
        case PROP_NUM_PROJECTIONS:
            g_value_set_uint (value, priv->num_projections);
            break;
        case PROP_COMPUTE_TYPE:
            g_value_set_enum (value, priv->compute_type);
            break;
        case PROP_RESULT_TYPE:
            g_value_set_enum (value, priv->result_type);
            break;
        case PROP_STORE_TYPE:
            g_value_set_enum (value, priv->store_type);
            break;
        case PROP_OVERALL_ANGLE:
            g_value_set_double (value, priv->overall_angle);
            break;
        case PROP_GRAY_MAP_MIN:
            g_value_set_double (value, priv->gray_map_min);
            break;
        case PROP_GRAY_MAP_MAX:
            g_value_set_double (value, priv->gray_map_max);
            break;
        case PROP_ADDRESSING_MODE:
            g_value_set_enum (value, priv->addressing_mode);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_general_backproject_task_finalize (GObject *object)
{
    guint i;
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (object);

    g_value_array_free (priv->region);
    g_value_array_free (priv->region_x);
    g_value_array_free (priv->region_y);
    g_value_array_free (priv->center_x);
    g_value_array_free (priv->center_z);
    g_value_array_free (priv->source_position_x);
    g_value_array_free (priv->source_position_y);
    g_value_array_free (priv->source_position_z);
    g_value_array_free (priv->detector_position_x);
    g_value_array_free (priv->detector_position_y);
    g_value_array_free (priv->detector_position_z);
    g_value_array_free (priv->detector_angle_x);
    g_value_array_free (priv->detector_angle_y);
    g_value_array_free (priv->detector_angle_z);
    g_value_array_free (priv->axis_angle_x);
    g_value_array_free (priv->axis_angle_y);
    g_value_array_free (priv->axis_angle_z);
    g_value_array_free (priv->volume_angle_x);
    g_value_array_free (priv->volume_angle_y);
    g_value_array_free (priv->volume_angle_z);

    for (i = 0; i < BURST; i++) {
        if (priv->projections[i] != NULL) {
            UFO_RESOURCES_CHECK_CLERR (clReleaseMemObject (priv->projections[i]));
            priv->projections[i] = NULL;
        }
    }

    if (priv->chunks) {
        for (i = 0; i < priv->num_chunks; i++) {
            UFO_RESOURCES_CHECK_CLERR (clReleaseMemObject (priv->chunks[i]));
        }
        g_free (priv->chunks);
        priv->chunks = NULL;
    }

    if (priv->cl_regions) {
        for (i = 0; i < priv->num_chunks; i++) {
            UFO_RESOURCES_CHECK_CLERR (clReleaseMemObject (priv->cl_regions[i]));
        }
        g_free (priv->cl_regions);
        priv->cl_regions = NULL;
    }

    if (priv->kernel) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->kernel));
        priv->kernel = NULL;
    }
    if (priv->rest_kernel) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->rest_kernel));
        priv->rest_kernel = NULL;
    }

    if (priv->context) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseContext (priv->context));
        priv->context = NULL;
    }
    if (priv->sampler) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseSampler (priv->sampler));
        priv->sampler = NULL;
    }

    G_OBJECT_CLASS (ufo_general_backproject_task_parent_class)->finalize (object);
}

static void
ufo_task_interface_init (UfoTaskIface *iface)
{
    iface->setup = ufo_general_backproject_task_setup;
    iface->get_num_inputs = ufo_general_backproject_task_get_num_inputs;
    iface->get_num_dimensions = ufo_general_backproject_task_get_num_dimensions;
    iface->get_mode = ufo_general_backproject_task_get_mode;
    iface->get_requisition = ufo_general_backproject_task_get_requisition;
    iface->process = ufo_general_backproject_task_process;
    iface->generate = ufo_general_backproject_task_generate;
}

static void
ufo_general_backproject_task_class_init (UfoGeneralBackprojectTaskClass *klass)
{
    GObjectClass *oclass = G_OBJECT_CLASS (klass);

    oclass->set_property = ufo_general_backproject_task_set_property;
    oclass->get_property = ufo_general_backproject_task_get_property;
    oclass->finalize = ufo_general_backproject_task_finalize;

    GParamSpec *region_vals = g_param_spec_int ("region-values",
                                                "Region values",
                                                "Elements in regions",
                                                G_MININT,
                                                G_MAXINT,
                                                (gint) 0,
                                                G_PARAM_READWRITE);

    GParamSpec *double_region_vals = g_param_spec_double ("double-region-values",
                                                          "Double Region values",
                                                          "Elements in double regions",
                                                          -G_MAXDOUBLE,
                                                          G_MAXDOUBLE,
                                                          0.0,
                                                          G_PARAM_READWRITE);

    properties[PROP_PARAMETER] =
        g_param_spec_enum ("parameter",
            "Which parameter will be varied along the z-axis",
            "Which parameter will be varied along the z-axis",
            g_enum_register_static ("GBPParameter", parameter_values),
            PARAMETER_Z,
            G_PARAM_READWRITE);

    properties[PROP_Z] =
        g_param_spec_double ("z",
            "Z coordinate of the reconstructed slice",
            "Z coordinate of the reconstructed slice",
            -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
            G_PARAM_READWRITE);

    properties[PROP_REGION] =
        g_param_spec_value_array ("region",
            "Region for the parameter along z-axis as (from, to, step)",
            "Region for the parameter along z-axis as (from, to, step)",
            double_region_vals,
            G_PARAM_READWRITE);

    properties[PROP_REGION_X] =
        g_param_spec_value_array ("x-region",
            "X region for reconstruction (horizontal axis) as (from, to, step)",
            "X region for reconstruction (horizontal axis) as (from, to, step)",
            region_vals,
            G_PARAM_READWRITE);

    properties[PROP_REGION_Y] =
        g_param_spec_value_array ("y-region",
            "Y region for reconstruction (beam direction axis) as (from, to, step)",
            "Y region for reconstruction (beam direction axis) as (from, to, step)",
            region_vals,
            G_PARAM_READWRITE);

    properties[PROP_CENTER_X] =
        g_param_spec_value_array ("center-x",
                                  "Global x center (horizontal in a projection) of the volume with respect to projections",
                                  "Global x center (horizontal in a projection) of the volume with respect to projections",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_CENTER_Z] =
        g_param_spec_value_array ("center-z",
                                  "Global z center (vertical in a projection) of the volume with respect to projections",
                                  "Global z center (vertical in a projection) of the volume with respect to projections",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_X] =
        g_param_spec_value_array ("source-position-x",
                                  "X source position (horizontal) in global coordinates [pixels]",
                                  "X source position (horizontal) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_Y] =
        g_param_spec_value_array ("source-position-y",
                                  "Y source position (beam direction) in global coordinates [pixels]",
                                  "Y source position (beam direction) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_Z] =
        g_param_spec_value_array ("source-position-z",
                                  "Z source position (vertical) in global coordinates [pixels]",
                                  "Z source position (vertical) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_X] =
        g_param_spec_value_array ("detector-position-x",
                                  "X detector position (horizontal) in global coordinates [pixels]",
                                  "X detector position (horizontal) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_Y] =
        g_param_spec_value_array ("detector-position-y",
                                  "Y detector position (along beam direction) in global coordinates [pixels]",
                                  "Y detector position (along beam direction) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_Z] =
        g_param_spec_value_array ("detector-position-z",
                                  "Z detector position (vertical) in global coordinates [pixels]",
                                  "Z detector position (vertical) in global coordinates [pixels]",
                                  double_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_X] =
        g_param_spec_value_array("detector-angle-x",
                                 "Detector rotation around the x axis [rad] (horizontal)",
                                 "Detector rotation around the x axis [rad] (horizontal)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_Y] =
        g_param_spec_value_array("detector-angle-y",
                                 "Detector rotation around the y axis [rad] (along beam direction)",
                                 "Detector rotation around the y axis [rad] (balong eam direction)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_Z] =
        g_param_spec_value_array("detector-angle-z",
                                 "Detector rotation around the z axis [rad] (vertical)",
                                 "Detector rotation around the z axis [rad] (vertical)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_X] =
        g_param_spec_value_array("axis-angle-x",
                                 "Rotation axis rotation around the x axis [rad] (laminographic angle, 0 = tomography)",
                                 "Rotation axis rotation around the x axis [rad] (laminographic angle, 0 = tomography)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_Y] =
        g_param_spec_value_array("axis-angle-y",
                                 "Rotation axis rotation around the y axis [rad] (along beam direction)",
                                 "Rotation axis rotation around the y axis [rad] (along beam direction)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_Z] =
        g_param_spec_value_array("axis-angle-z",
                                 "Rotation axis rotation around the z axis [rad] (vertical)",
                                 "Rotation axis rotation around the z axis [rad] (vertical)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_X] =
        g_param_spec_value_array("volume-angle-x",
                                 "Volume rotation around the x axis [rad] (horizontal)",
                                 "Volume rotation around the x axis [rad] (horizontal)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_Y] =
        g_param_spec_value_array("volume-angle-y",
                                 "Volume rotation around the y axis [rad] (along beam direction)",
                                 "Volume rotation around the y axis [rad] (along beam direction)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_Z] =
        g_param_spec_value_array("volume-angle-z",
                                 "Volume rotation around the z axis [rad] (vertical)",
                                 "Volume rotation around the z axis [rad] (vertical)",
                                 double_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_COMPUTE_TYPE] =
        g_param_spec_enum ("compute-type",
                             "Data type for performing kernel math operations",
                             "Data type for performing kernel math operations "
                             "(\"half\", \"float\", \"double\")",
                             g_enum_register_static ("compute-type", compute_type_values),
                             FT_FLOAT,
                             G_PARAM_READWRITE);

    properties[PROP_ADDRESSING_MODE] =
        g_param_spec_enum ("addressing-mode",
            "Outlier treatment (\"none\", \"clamp\", \"clamp_to_edge\", \"repeat\")",
            "Outlier treatment (\"none\", \"clamp\", \"clamp_to_edge\", \"repeat\")",
            g_enum_register_static ("bp_addressing_mode", addressing_values),
            CL_ADDRESS_CLAMP,
            G_PARAM_READWRITE);

    properties[PROP_RESULT_TYPE] =
        g_param_spec_enum ("result-type",
                             "Data type for storing the intermediate gray value for a voxel from various rotation angles",
                             "Data type for storing the intermediate gray value for a voxel from various rotation angles "
                             "(\"half\", \"float\", \"double\")",
                             g_enum_register_static ("result-type", ft_values),
                             FT_FLOAT,
                             G_PARAM_READWRITE);

    properties[PROP_STORE_TYPE] =
        g_param_spec_enum ("store-type",
                             "Data type of the output volume",
                             "Data type of the output volume "
                             "(\"half\", \"float\", \"double\", \"uchar\", \"ushort\", \"uint\")",
                             g_enum_register_static ("store-type", st_values),
                             ST_FLOAT,
                             G_PARAM_READWRITE);

    properties[PROP_OVERALL_ANGLE] =
        g_param_spec_double ("overall-angle",
            "Angle covered by all projections [rad]",
            "Angle covered by all projections [rad] (can be negative for negative steps "
            "in case only num-projections is specified",
            -G_MAXDOUBLE, G_MAXDOUBLE, 2 * G_PI,
            G_PARAM_READWRITE);

    properties[PROP_GRAY_MAP_MIN] =
        g_param_spec_double ("gray-map-min",
            "Gray valye which maps to 0 in case of integer store type",
            "Gray valye which maps to 0 in case of integer store type",
            -G_MAXDOUBLE, G_MAXDOUBLE, 0,
            G_PARAM_READWRITE);

    properties[PROP_GRAY_MAP_MAX] =
        g_param_spec_double ("gray-map-max",
            "Gray valye which maps to maximum of the chosen integer type in case of integer store type",
            "Gray valye which maps to maximum of the chosen integer type in case of integer store type",
            -G_MAXDOUBLE, G_MAXDOUBLE, 0,
            G_PARAM_READWRITE);

    properties[PROP_NUM_PROJECTIONS] =
        g_param_spec_uint ("num-projections",
            "Number of projections",
            "Number of projections",
            0, 16384, 0,
            G_PARAM_READWRITE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (oclass, i, properties[i]);

    g_type_class_add_private (oclass, sizeof(UfoGeneralBackprojectTaskPrivate));
}

static void
ufo_general_backproject_task_init(UfoGeneralBackprojectTask *self)
{
    gint i;
    self->priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE(self);
    GValue double_value = G_VALUE_INIT;
    GValue int_value = G_VALUE_INIT;
    g_value_init (&double_value, G_TYPE_DOUBLE);
    g_value_init (&int_value, G_TYPE_INT);
    g_value_set_double (&double_value, 0.0);
    g_value_set_int (&int_value, 0);

    /* Scalars */
    self->priv->parameter = PARAMETER_Z;
    self->priv->z = 0.0;
    self->priv->num_projections = 0;
    self->priv->compute_type = CT_FLOAT;
    self->priv->result_type = FT_FLOAT;
    self->priv->store_type = ST_FLOAT;
    self->priv->overall_angle = 2 * G_PI;
    self->priv->addressing_mode = CL_ADDRESS_CLAMP;
    self->priv->gray_map_min = 0.0;
    self->priv->gray_map_max = 0.0;

    /* Value arrays */
    self->priv->region = g_value_array_new (3);
    self->priv->region_x = g_value_array_new (3);
    self->priv->region_y = g_value_array_new (3);
    self->priv->center_x = g_value_array_new (1);
    self->priv->center_z = g_value_array_new (1);
    self->priv->axis_angle_x = g_value_array_new (1);
    self->priv->axis_angle_y = g_value_array_new (1);
    self->priv->axis_angle_z = g_value_array_new (1);
    self->priv->volume_angle_x = g_value_array_new (1);
    self->priv->volume_angle_y = g_value_array_new (1);
    self->priv->volume_angle_z = g_value_array_new (1);
    self->priv->source_position_x = g_value_array_new (1);
    self->priv->source_position_y = g_value_array_new (1);
    self->priv->source_position_z = g_value_array_new (1);
    self->priv->detector_position_x = g_value_array_new (1);
    self->priv->detector_position_y = g_value_array_new (1);
    self->priv->detector_position_z = g_value_array_new (1);
    self->priv->detector_angle_x = g_value_array_new (1);
    self->priv->detector_angle_y = g_value_array_new (1);
    self->priv->detector_angle_z = g_value_array_new (1);

    for (i = 0; i < 3; i++) {
        g_value_array_insert (self->priv->region, i, &double_value);
        g_value_array_insert (self->priv->region_x, i, &int_value);
        g_value_array_insert (self->priv->region_y, i, &int_value);
    }
    g_value_array_insert (self->priv->center_x, 0, &double_value);
    g_value_array_insert (self->priv->center_z, 0, &double_value);

    g_value_array_insert (self->priv->axis_angle_x, 0, &double_value);
    g_value_array_insert (self->priv->axis_angle_y, 0, &double_value);
    g_value_array_insert (self->priv->axis_angle_z, 0, &double_value);

    g_value_array_insert (self->priv->volume_angle_x, 0, &double_value);
    g_value_array_insert (self->priv->volume_angle_y, 0, &double_value);
    g_value_array_insert (self->priv->volume_angle_z, 0, &double_value);

    g_value_array_insert (self->priv->source_position_x, 0, &double_value);
    g_value_array_insert (self->priv->source_position_z, 0, &double_value);
    g_value_set_double (&double_value, -INFINITY);
    g_value_array_insert (self->priv->source_position_y, 0, &double_value);

    g_value_set_double (&double_value, 0.0f);
    g_value_array_insert (self->priv->detector_position_x, 0, &double_value);
    g_value_array_insert (self->priv->detector_position_y, 0, &double_value);
    g_value_array_insert (self->priv->detector_position_z, 0, &double_value);
    g_value_array_insert (self->priv->detector_angle_x, 0, &double_value);
    g_value_array_insert (self->priv->detector_angle_z, 0, &double_value);
    g_value_array_insert (self->priv->detector_angle_y, 0, &double_value);

    /* Private */
    self->priv->num_slices = 0;
    self->priv->num_slices_per_chunk = 0;
    self->priv->count = 0;
    self->priv->generated = 0;
}
/*}}}*/

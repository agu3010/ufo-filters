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
#include "ufo-general-backproject-task.h"

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
    GValueArray *region, *region_x, *region_y;
    GValueArray *center_x, *center_z;
    GValueArray *source_position_x, *source_position_y, *source_position_z;
    GValueArray *detector_position_x, *detector_position_y, *detector_position_z;
    GValueArray *detector_angle_x, *detector_angle_y, *detector_angle_z;
    GValueArray *axis_angle_x, *axis_angle_y, *axis_angle_z;
    GValueArray *volume_angle_x, *volume_angle_y, *volume_angle_z;
    FloatType compute_type, result_type;
    StoreType store_type;
    Parameter parameter;
    /* Private */
    guint count;
    gfloat sines[BURST], cosines[BURST];
    guint num_projections;
    gfloat overall_angle;
    gboolean generated;
    /* OpenCL */
    cl_context context;
    cl_kernel kernel, rest_kernel;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoGeneralBackprojectTask, ufo_general_backproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_GENERAL_BACKPROJECT_TASK, UfoGeneralBackprojectTaskPrivate))

enum {
    PROP_0,
    PROP_PARAMETER,
    PROP_REGION,
    PROP_region_x,
    PROP_region_y,
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
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

/*{{{ General helper functions*/
static gboolean
are_almost_equal (gfloat a, gfloat b)
{
    return (fabs (a - b) < 1e-7);
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
    gint is_uchar = !g_strcmp0 (store_type, "uchar");
    gint is_ushort = !g_strcmp0 (store_type, "ushort");
    gint is_uint = !g_strcmp0 (store_type, "uint");
    guint maxval = 0;

    if (is_uchar) {
        maxval = 0xFF;
    } else if (is_ushort) {
        maxval = 0xFFFF;
    } else if (is_uint) {
        maxval = 0xFFFFFFFF;
    }

    if (maxval) {
        written = g_snprintf (code, size,
                              "(%s) clamp ((%s)(gray_limit.y * (norm_factor * result - gray_limit.x)), (%s) 0.0, (%s) %u.0)",
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
    gboolean with_axis, with_volume, parallel_beam, perpendicular_detector;
    gchar *template, *kernel_code;
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);

    /* Check parameter values */
    if (!priv->num_projections) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_SETUP,
                     "Number of projections not set");
        return;
    }

    /* Assume the most efficient geometry, change if necessary */
    with_axis = FALSE;
    with_volume = FALSE;
    perpendicular_detector = TRUE;
    parallel_beam = TRUE;
    for (i = 0; i < priv->num_projections; i++) {
        if (!(are_almost_equal (get_float_from_array_or_scalar (priv->axis_angle_x, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->axis_angle_y, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->axis_angle_z, 0), 0))) {
            with_axis = TRUE;
        }
        if (!(are_almost_equal (get_float_from_array_or_scalar (priv->volume_angle_x, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->volume_angle_y, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->volume_angle_z, 0), 0))) {
            with_volume = TRUE;
        }
        if (!(are_almost_equal (get_float_from_array_or_scalar (priv->detector_angle_x, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->detector_angle_y, 0), 0) &&
              are_almost_equal (get_float_from_array_or_scalar (priv->detector_angle_z, 0), 0))) {
            perpendicular_detector = FALSE;
        }
        if (!isinf (get_float_from_array_or_scalar (priv->source_position_y, i))) {
            parallel_beam = FALSE;
        }
    }

    g_debug ("burst: %d, parameter: %s with axis: %d, with volume: %d, "
             "perpendicular detector: %d, parallel beam: %d, "
             "compute type: %s, result type: %s, store type: %s",
             BURST, parameter_values[priv->parameter].value_nick, with_axis, with_volume,
             perpendicular_detector, parallel_beam,
             ft_values[priv->compute_type].value_nick,
             ft_values[priv->result_type].value_nick,
             st_values[priv->store_type].value_nick);

    /* Create kernel source code based on geometry settings */
    if (!(template = ufo_resources_get_kernel_source (resources, "general_backproject.in", error))) {
        return;
    }

    kernel_code = make_kernel (template, BURST, with_axis, with_volume,
                               perpendicular_detector, parallel_beam,
                               ft_values[priv->compute_type].value_nick,
                               ft_values[priv->result_type].value_nick,
                               st_values[priv->store_type].value_nick,
                               parameter_values[priv->parameter].value_nick, error);
    priv->kernel = ufo_resources_get_kernel_from_source_with_opts (resources,
                                                                   kernel_code,
                                                                   "backproject",
                                                                   NULL,
                                                                   error);
    g_free (kernel_code);

    kernel_code = make_kernel (template, priv->num_projections % BURST,
                               with_axis, with_volume,
                               perpendicular_detector, parallel_beam,
                               ft_values[priv->compute_type].value_nick,
                               ft_values[priv->result_type].value_nick,
                               st_values[priv->store_type].value_nick,
                               parameter_values[priv->parameter].value_nick, error);

    /* If num_projections % BURST != 0 we need one more kernel to process the remaining projections */
    priv->rest_kernel = ufo_resources_get_kernel_from_source_with_opts (resources,
                                                                        kernel_code,
                                                                        "backproject",
                                                                        NULL,
                                                                        error);

    /* Set OpenCL variables */
    priv->context = ufo_resources_get_context (resources);
    UFO_RESOURCES_CHECK_CLERR (clRetainContext (priv->context));

    if (priv->kernel) {
        UFO_RESOURCES_CHECK_CLERR (clRetainKernel (priv->kernel));
    }
    if (priv->rest_kernel) {
        UFO_RESOURCES_CHECK_CLERR (clRetainKernel (priv->rest_kernel));
    }

    g_free (template);
    g_free (kernel_code);
}

static void
ufo_general_backproject_task_get_requisition (UfoTask *task,
                                 UfoBuffer **inputs,
                                 UfoRequisition *requisition)
{
    UfoGeneralBackprojectTaskPrivate *priv;
    UfoRequisition in_req;
    gfloat start, stop, step;
    GValue g_value_int = G_VALUE_INIT;
    g_value_init (&g_value_int, G_TYPE_INT);

    priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);
    g_assert (priv->region->n_values == 3);
    requisition->n_dims = 3;
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
    if (are_almost_equal (EXTRACT_FLOAT (priv->region, 2), 0.0f)) {
        /* Conservative approach, reconstruct just one slice */
        start = 0.0f;
        stop = 1.0f;
        step = 1.0f;
    } else {
        start = EXTRACT_FLOAT (priv->region, 0);
        stop = EXTRACT_FLOAT (priv->region, 1);
        step = EXTRACT_FLOAT (priv->region, 2);
    }

    requisition->dims[2] = (gint) ceil ((stop - start) / step);

    g_debug ("requisition (x, y, z): %lu %lu %lu", requisition->dims[0],
             requisition->dims[1], requisition->dims[2]);
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

    return 3;
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
    return TRUE;
}

static gboolean
ufo_general_backproject_task_generate (UfoTask *task,
                         UfoBuffer *output,
                         UfoRequisition *requisition)
{
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (task);

    if (priv->generated) {
        return FALSE;
    }

    priv->generated = TRUE;

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
        case PROP_REGION:
            array = (GValueArray *) g_value_get_boxed (value);
            g_value_array_free (priv->region);
            priv->region = g_value_array_copy (array);
            break;
        case PROP_region_x:
            array = (GValueArray *) g_value_get_boxed (value);
            set_region (array, &priv->region_x);
            break;
        case PROP_region_y:
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
            priv->overall_angle = g_value_get_float (value);
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
        case PROP_REGION:
            g_value_set_boxed (value, priv->region);
            break;
        case PROP_region_x:
            g_value_set_boxed (value, priv->region_x);
            break;
        case PROP_region_y:
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
            g_value_set_float (value, priv->overall_angle);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_general_backproject_task_finalize (GObject *object)
{
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

    GParamSpec *float_region_vals = g_param_spec_float ("float-region-values",
                                                        "Float Region values",
                                                        "Elements in float regions",
                                                        -INFINITY,
                                                        INFINITY,
                                                        0.0f,
                                                        G_PARAM_READWRITE);

    properties[PROP_PARAMETER] =
        g_param_spec_enum ("parameter",
            "Which parameter will be varied along the z-axis",
            "Which parameter will be varied along the z-axis",
            g_enum_register_static ("GBPParameter", parameter_values),
            PARAMETER_Z,
            G_PARAM_READWRITE);

    properties[PROP_REGION] =
        g_param_spec_value_array ("region",
            "Region for the parameter along z-axis as (from, to, step)",
            "Region for the parameter along z-axis as (from, to, step)",
            float_region_vals,
            G_PARAM_READWRITE);

    properties[PROP_region_x] =
        g_param_spec_value_array ("x-region",
            "X region for reconstruction (horizontal axis) as (from, to, step)",
            "X region for reconstruction (horizontal axis) as (from, to, step)",
            region_vals,
            G_PARAM_READWRITE);

    properties[PROP_region_y] =
        g_param_spec_value_array ("y-region",
            "Y region for reconstruction (beam direction axis) as (from, to, step)",
            "Y region for reconstruction (beam direction axis) as (from, to, step)",
            region_vals,
            G_PARAM_READWRITE);

    properties[PROP_CENTER_X] =
        g_param_spec_value_array ("center-x",
                                  "Global x center (horizontal in a projection) of the volume with respect to projections",
                                  "Global x center (horizontal in a projection) of the volume with respect to projections",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_CENTER_Z] =
        g_param_spec_value_array ("center-z",
                                  "Global z center (vertical in a projection) of the volume with respect to projections",
                                  "Global z center (vertical in a projection) of the volume with respect to projections",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_X] =
        g_param_spec_value_array ("source-position-x",
                                  "X source position (horizontal) in global coordinates [pixels]",
                                  "X source position (horizontal) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_Y] =
        g_param_spec_value_array ("source-position-y",
                                  "Y source position (beam direction) in global coordinates [pixels]",
                                  "Y source position (beam direction) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_SOURCE_POSITION_Z] =
        g_param_spec_value_array ("source-position-z",
                                  "Z source position (vertical) in global coordinates [pixels]",
                                  "Z source position (vertical) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_X] =
        g_param_spec_value_array ("detector-position-x",
                                  "X detector position (horizontal) in global coordinates [pixels]",
                                  "X detector position (horizontal) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_Y] =
        g_param_spec_value_array ("detector-position-y",
                                  "Y detector position (along beam direction) in global coordinates [pixels]",
                                  "Y detector position (along beam direction) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_POSITION_Z] =
        g_param_spec_value_array ("detector-position-z",
                                  "Z detector position (vertical) in global coordinates [pixels]",
                                  "Z detector position (vertical) in global coordinates [pixels]",
                                  float_region_vals,
                                  G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_X] =
        g_param_spec_value_array("detector-angle-x",
                                 "Detector rotation around the x axis [rad] (horizontal)",
                                 "Detector rotation around the x axis [rad] (horizontal)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_Y] =
        g_param_spec_value_array("detector-angle-y",
                                 "Detector rotation around the y axis [rad] (along beam direction)",
                                 "Detector rotation around the y axis [rad] (balong eam direction)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_DETECTOR_ANGLE_Z] =
        g_param_spec_value_array("detector-angle-z",
                                 "Detector rotation around the z axis [rad] (vertical)",
                                 "Detector rotation around the z axis [rad] (vertical)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_X] =
        g_param_spec_value_array("axis-angle-x",
                                 "Rotation axis rotation around the x axis [rad] (laminographic angle, 0 = tomography)",
                                 "Rotation axis rotation around the x axis [rad] (laminographic angle, 0 = tomography)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_Y] =
        g_param_spec_value_array("axis-angle-y",
                                 "Rotation axis rotation around the y axis [rad] (along beam direction)",
                                 "Rotation axis rotation around the y axis [rad] (along beam direction)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_AXIS_ANGLE_Z] =
        g_param_spec_value_array("axis-angle-z",
                                 "Rotation axis rotation around the z axis [rad] (vertical)",
                                 "Rotation axis rotation around the z axis [rad] (vertical)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_X] =
        g_param_spec_value_array("volume-angle-x",
                                 "Volume rotation around the x axis [rad] (horizontal)",
                                 "Volume rotation around the x axis [rad] (horizontal)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_Y] =
        g_param_spec_value_array("volume-angle-y",
                                 "Volume rotation around the y axis [rad] (along beam direction)",
                                 "Volume rotation around the y axis [rad] (along beam direction)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_VOLUME_ANGLE_Z] =
        g_param_spec_value_array("volume-angle-z",
                                 "Volume rotation around the z axis [rad] (vertical)",
                                 "Volume rotation around the z axis [rad] (vertical)",
                                 float_region_vals,
                                 G_PARAM_READWRITE);

    properties[PROP_COMPUTE_TYPE] =
        g_param_spec_enum ("compute-type",
                             "Data type for performing kernel math operations",
                             "Data type for performing kernel math operations "
                             "(\"half\", \"float\", \"double\")",
                             g_enum_register_static ("compute-type", ft_values),
                             FT_FLOAT,
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
        g_param_spec_float ("overall-angle",
            "Angle covered by all projections [rad]",
            "Angle covered by all projections [rad] (can be negative for negative steps "
            "in case only num-projections is specified",
            -G_MAXFLOAT, G_MAXFLOAT, 2 * G_PI,
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
    GValue float_value = G_VALUE_INIT;
    GValue int_value = G_VALUE_INIT;
    g_value_init (&float_value, G_TYPE_FLOAT);
    g_value_init (&int_value, G_TYPE_INT);
    g_value_set_float (&float_value, 0.0f);
    g_value_set_int (&int_value, 0);

    /* Scalars */
    self->priv->parameter = PARAMETER_Z;
    self->priv->num_projections = 0;
    self->priv->compute_type = FT_FLOAT;
    self->priv->result_type = FT_FLOAT;
    self->priv->store_type = FT_FLOAT;
    self->priv->overall_angle = 2 * G_PI;

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
        g_value_array_insert (self->priv->region, i, &float_value);
        g_value_array_insert (self->priv->region_x, i, &int_value);
        g_value_array_insert (self->priv->region_y, i, &int_value);
    }
    g_value_array_insert (self->priv->center_x, 0, &float_value);
    g_value_array_insert (self->priv->center_z, 0, &float_value);

    g_value_array_insert (self->priv->axis_angle_x, 0, &float_value);
    g_value_array_insert (self->priv->axis_angle_y, 0, &float_value);
    g_value_array_insert (self->priv->axis_angle_z, 0, &float_value);

    g_value_array_insert (self->priv->volume_angle_x, 0, &float_value);
    g_value_array_insert (self->priv->volume_angle_y, 0, &float_value);
    g_value_array_insert (self->priv->volume_angle_z, 0, &float_value);

    g_value_array_insert (self->priv->source_position_x, 0, &float_value);
    g_value_array_insert (self->priv->source_position_z, 0, &float_value);
    g_value_set_float (&float_value, -INFINITY);
    g_value_array_insert (self->priv->source_position_y, 0, &float_value);

    g_value_set_float (&float_value, 0.0f);
    g_value_array_insert (self->priv->detector_position_x, 0, &float_value);
    g_value_array_insert (self->priv->detector_position_y, 0, &float_value);
    g_value_array_insert (self->priv->detector_position_z, 0, &float_value);
    g_value_array_insert (self->priv->detector_angle_x, 0, &float_value);
    g_value_array_insert (self->priv->detector_angle_z, 0, &float_value);
    g_value_array_insert (self->priv->detector_angle_y, 0, &float_value);

    /* Private */
    self->priv->count = 0;
    self->priv->generated = FALSE;
}
/*}}}*/

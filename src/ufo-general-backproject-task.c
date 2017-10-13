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

#include "ufo-general-backproject-task.h"
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


struct _UfoGeneralBackprojectTaskPrivate {
    gboolean foo;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoGeneralBackprojectTask, ufo_general_backproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_GENERAL_BACKPROJECT_TASK, UfoGeneralBackprojectTaskPrivate))

enum {
    PROP_0,
    PROP_TEST,
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

/*{{{ Kernel creation*/
static gchar *
read_file (const gchar *filename)
{
    gssize length;
    gssize buffer_length;
    FILE *fp = fopen (filename, "r");
    gchar *buffer;

    if (fp == NULL) {
        return NULL;
    }

    fseek (fp, 0, SEEK_END);
    length = (gsize) ftell (fp);

    if (length < 0) {
        return NULL;
    }

    rewind (fp);
    buffer = g_strnfill (length + 1, 0);

    if (buffer == NULL) {
        fclose (fp);
        return NULL;
    }

    buffer_length = fread (buffer, 1, length, fp);
    fclose (fp);

    if (buffer_length != length) {
        g_free (buffer);
        return NULL;
    }

    return buffer;
}

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

static gchar *
make_kernel (gint burst, gboolean with_axis, gboolean with_volume, gboolean perpendicular_detector,
             gboolean parallel_beam, const gchar *compute_type, const gchar *result_type,
             const gchar *store_type, const gchar *parameter)
{
    const gchar *double_pragma_def, *double_pragma, *half_pragma_def, *half_pragma,
          *image_args_fmt, *trigonomoerty_args_fmt;
    gchar *tmpl, *image_args, *trigonometry_args, *type_conversion, *parameter_assignment,
          *static_transformations, *transformations, *code_tmp, *code, **parts;

    double_pragma_def = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    half_pragma_def = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n";
    image_args_fmt = "\t\t\t read_only image2d_t projection_%02d,\n";
    trigonomoerty_args_fmt = "\t\t\t const cfloat2 tomo_%02d,\n";
    tmpl = read_file ("template.in");
    parts = g_strsplit (tmpl, "%tmpl%", 8);

    if ((image_args = make_args (burst, image_args_fmt)) == NULL) {
        g_warning ("Error making image arguments");
        return NULL;
    }
    if ((trigonometry_args = make_args (burst, trigonomoerty_args_fmt)) == NULL) {
        g_warning ("Error making trigonometric arguments");
        return NULL;
    }
    if ((type_conversion = make_type_conversion (compute_type, store_type)) == NULL) {
        g_warning ("Error making type conversion");
        return NULL;
    }
    parameter_assignment = make_parameter_assignment (parameter);
    if (parameter_assignment == NULL) {
        g_warning ("Wrong parameter name");
        return NULL;
    }

    if ((static_transformations = make_static_transformations(with_volume, perpendicular_detector,
                                                              parallel_beam)) == NULL) {
        g_warning ("Error making static transformations");
        return NULL;
    }
    if ((transformations = make_transformations (burst, with_axis, perpendicular_detector,
                                            parallel_beam, compute_type)) == NULL) {
        g_warning ("Wrong parameter name");
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

    g_free (tmpl);
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
}

static void
ufo_general_backproject_task_get_requisition (UfoTask *task,
                                 UfoBuffer **inputs,
                                 UfoRequisition *requisition)
{
    requisition->n_dims = 0;
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
    return TRUE;
}

static gboolean
ufo_general_backproject_task_generate (UfoTask *task,
                         UfoBuffer *output,
                         UfoRequisition *requisition)
{
    return TRUE;
}

static void
ufo_general_backproject_task_set_property (GObject *object,
                              guint property_id,
                              const GValue *value,
                              GParamSpec *pspec)
{
    UfoGeneralBackprojectTaskPrivate *priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_TEST:
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
        case PROP_TEST:
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_general_backproject_task_finalize (GObject *object)
{
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

    properties[PROP_TEST] =
        g_param_spec_string ("test",
            "Test property nick",
            "Test property description blurb",
            "",
            G_PARAM_READWRITE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (oclass, i, properties[i]);

    g_type_class_add_private (oclass, sizeof(UfoGeneralBackprojectTaskPrivate));
}

static void
ufo_general_backproject_task_init(UfoGeneralBackprojectTask *self)
{
    self->priv = UFO_GENERAL_BACKPROJECT_TASK_GET_PRIVATE(self);
}

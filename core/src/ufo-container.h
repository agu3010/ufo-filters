#ifndef __UFO_CONTAINER_H
#define __UFO_CONTAINER_H

#include <glib-object.h>
#include "ufo-element.h"

G_BEGIN_DECLS

#define UFO_TYPE_CONTAINER             (ufo_container_get_type())
#define UFO_CONTAINER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), UFO_TYPE_CONTAINER, UfoContainer))
#define UFO_IS_CONTAINER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), UFO_TYPE_BUFFER))
#define UFO_CONTAINER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), UFO_TYPE_CONTAINER, UfoContainerClass))
#define UFO_IS_CONTAINER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), UFO_TYPE_BUFFER))
#define UFO_CONTAINER_GET_CLASS(obj)   (G_TYPE_INSTANCE_GET_CLASS((obj), UFO_TYPE_CONTAINER, UfoContainerClass))

typedef struct _UfoContainer           UfoContainer;
typedef struct _UfoContainerClass      UfoContainerClass;
typedef struct _UfoContainerPrivate    UfoContainerPrivate;

struct _UfoContainer {
    UfoElement parent_instance;

    /* public */

    /* private */
    UfoContainerPrivate *priv;
};

struct _UfoContainerClass {
    UfoElementClass parent_class;

    /* class members */

    /* virtual public methods */
    void (*add_element) (UfoContainer *container, UfoElement *element); 
};

/* virtual public methods */

/* non-virtual public methods */
UfoContainer *ufo_container_new();
void ufo_container_add_element(UfoContainer *container, UfoElement *element);

GType ufo_container_get_type(void);

G_END_DECLS

#endif
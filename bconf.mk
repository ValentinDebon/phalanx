CFLAGS+=-std=c11
CPPFLAGS+=-D_GNU_SOURCE

llama-CPPFLAGS:=$(shell pkg-config --cflags-only-I llama)
llama-CFLAGS:=$(shell pkg-config --cflags-only-other llama)
llama-LDFLAGS:=$(shell pkg-config --libs-only-L llama)
llama-LDLIBS:=$(shell pkg-config --libs-only-l llama)

phalanx-objs:=src/phalanx.o

src/phalanx.o: CPPFLAGS+=$(llama-CPPFLAGS) \
	-DCONFIG_LARGE_LANGUAGE_MODEL_PATH='"$(CONFIG_LARGE_LANGUAGE_MODEL_PATH)"'

phalanx: LDFLAGS+=$(llama-LDFLAGS)
phalanx: LDLIBS+=$(llama-LDLIBS)
phalanx: $(phalanx-objs)

host-bin+=phalanx
clean-up+=$(host-bin) $(phalanx-objs)

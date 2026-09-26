#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstdlib>
#include <intrin.h>

#ifndef __attribute__
#define __attribute__(unused)
#endif
#ifndef __builtin_expect
#define __builtin_expect(value, expected) (value)
#endif
#ifndef __builtin_trap
#define __builtin_trap() std::abort()
#endif
#ifndef __builtin_assume_aligned
#define __builtin_assume_aligned(ptr, alignment) (ptr)
#endif
#endif

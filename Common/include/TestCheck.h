#ifndef NLF_TEST_CHECK_H
#define NLF_TEST_CHECK_H

/**
 * TestCheck.h — assertion macro for programs registered as CTest cases.
 *
 * Use NLF_CHECK, never assert(). Release builds define NDEBUG, which compiles
 * assert() out entirely: a test built that way still exits 0 no matter what it
 * computed, so CTest reports it green while it verifies nothing. NLF_CHECK is a
 * plain `if` and survives every build type.
 *
 * A CTest case passes or fails on its EXIT CODE alone. Printing "FAIL" or
 * "WARNING" and then returning 0 is not a test -- CTest hides stdout by default,
 * so the diagnosis is discarded and the case reports green. Anything registered
 * via add_test() must therefore return non-zero when its invariants break, which
 * is what this macro does.
 *
 * Only usable inside a function returning int (i.e. main()).
 */

#include <cstdio>

#define NLF_CHECK(cond, msg)                                                   \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL: %s\n    (%s)\n    at %s:%d\n",         \
                         (msg), #cond, __FILE__, __LINE__);                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#endif  // NLF_TEST_CHECK_H

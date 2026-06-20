/**
 * @file   test_mp.cpp
 * @brief  Runner for the distributed (Phase 3) test suite.
 *
 * GTest and MPI don't mix cleanly, so this is a minimal stand-in: owns the
 * MP_TEST registry (see mp_test.h) and main(), and runs every test case
 * registered by the other test_*_mp.cpp translation units in this binary.
 * Each test prints its own PASS/FAIL line; this file just aggregates the
 * final tally.
 *
 * Run:  srun -N<nodes> --tasks-per-node=<p> --gpus-per-node=<p> build/cuTestMp [nprow npcol] [nb]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "mp_test.h"
#include <chrono>
#include <cstdlib>
#include <map>

namespace mptest {
std::vector<TestCase> &registry() {
    static std::vector<TestCase> r;
    return r;
}
} // namespace mptest

using Clock = std::chrono::steady_clock;
static long long ms_since(Clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int nprow = 0, npcol = 0, nb = 256;
    if (argc >= 3) {
        nprow = atoi(argv[1]);
        npcol = atoi(argv[2]);
    } else {
        cuev::mp::grid_factor(world, nprow, npcol);
    }
    if (argc >= 4) nb = atoi(argv[3]);

    cuev::mp::Context ctx;
    cuev::mp::ctx_init(ctx, nb, nprow, npcol);

    auto &tests = mptest::registry();

    // Group by suite, preserving first-seen order (mirrors GTest's grouping).
    std::vector<std::string> suite_order;
    std::map<std::string, std::vector<const mptest::TestCase *>> by_suite;
    for (auto &tc : tests) {
        if (by_suite.find(tc.suite) == by_suite.end()) suite_order.push_back(tc.suite);
        by_suite[tc.suite].push_back(&tc);
    }

    const char *green = mptest::color::green();
    const char *red = mptest::color::red();
    const char *reset = mptest::color::reset();

    if (ctx.rank == 0)
        printf("%s[==========]%s Running %zu tests from %zu test suites.\n", green, reset,
               tests.size(), suite_order.size());

    std::vector<std::string> failed_names;
    auto t_all = Clock::now();

    for (auto &suite : suite_order) {
        auto &cases = by_suite[suite];
        if (ctx.rank == 0)
            printf("%s[----------]%s %zu test%s from %s\n", green, reset, cases.size(),
                   cases.size() == 1 ? "" : "s", suite.c_str());

        auto t_suite = Clock::now();
        for (auto *tc : cases) {
            if (ctx.rank == 0) printf("%s[ RUN      ]%s %s\n", green, reset, tc->name.c_str());

            auto t_case = Clock::now();
            bool pass = tc->fn(ctx);
            long long elapsed = ms_since(t_case);

            if (ctx.rank == 0) {
                if (pass)
                    printf("%s[       OK ]%s %s (%lld ms)\n", green, reset, tc->name.c_str(),
                           elapsed);
                else
                    printf("%s[  FAILED  ]%s %s (%lld ms)\n", red, reset, tc->name.c_str(),
                           elapsed);
            }
            if (!pass) failed_names.push_back(tc->name);
        }
        if (ctx.rank == 0)
            printf("%s[----------]%s %zu test%s from %s (%lld ms total)\n\n", green, reset,
                   cases.size(), cases.size() == 1 ? "" : "s", suite.c_str(), ms_since(t_suite));
    }

    if (ctx.rank == 0) {
        printf("%s[==========]%s %zu tests from %zu test suites ran. (%lld ms total)\n", green,
               reset, tests.size(), suite_order.size(), ms_since(t_all));
        size_t passed = tests.size() - failed_names.size();
        printf("%s[  PASSED  ]%s %zu test%s.\n", green, reset, passed, passed == 1 ? "" : "s");
        if (!failed_names.empty()) {
            printf("%s[  FAILED  ]%s %zu test%s, listed below:\n", red, reset, failed_names.size(),
                   failed_names.size() == 1 ? "" : "s");
            for (auto &n : failed_names)
                printf("%s[  FAILED  ]%s %s\n", red, reset, n.c_str());
        }
    }

    int fails = (int)failed_names.size();
    cuev::mp::ctx_finalize(ctx);
    MPI_Finalize();
    return fails ? 1 : 0;
}

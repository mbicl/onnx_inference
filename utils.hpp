#include <chrono>

namespace Utils {
    template <typename Clock = std::chrono::high_resolution_clock>
    class StopWatch {
        typename Clock::time_point start;
    public:
        StopWatch() : start(Clock::now()) {}

        template<typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
        Rep elapsed() const {
            std::atomic_thread_fence(std::memory_order_relaxed);
            auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start).count();
            std::atomic_thread_fence(std::memory_order_relaxed);
            return static_cast<Rep>(counted_time);
        }
    };
}
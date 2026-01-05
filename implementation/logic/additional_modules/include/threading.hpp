#pragma once
#include <vector>
#include <thread>
#include <functional>
namespace additional_modules::threading
{
    template <typename Func, typename... Args>
    void split_to_threads(int rows, int n_threads, Func &&func, Args &&...args)
    {
        if (n_threads <= 0)
            n_threads = 1;

        int chunk_size = rows / n_threads;
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        for (int t = 0; t < n_threads; ++t)
        {
            int start = t * chunk_size;
            int end = (t == n_threads - 1) ? rows : start + chunk_size;

            threads.emplace_back(std::forward<Func>(func),
                                 std::forward<Args>(args)...,
                                 start, end);
        }

        for (auto &t : threads)
        {
            if (t.joinable())
                t.join();
        }
    }
}
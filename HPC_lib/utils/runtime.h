// https://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows

#ifndef HPC_UTILS_RUNTIME_H
#define HPC_UTILS_RUNTIME_H


//Windows 
#ifdef _WIN32
#include <Windows.h>
double get_walltime(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cputime(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

// POSIX/Linux
#else 
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
double get_walltime(){
    struct tms ts;
    static double ClockTick=0;

    if (ClockTick==0.0) {
        ClockTick = 1.0 / ((double) sysconf(_SC_CLK_TCK));
    }
    return ((double) times(&ts)) * ClockTick;
}
double get_cputime(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif // end _WIN32 & POSIX/Linux


#endif // end HPC_UTILS_RUNTIME_H
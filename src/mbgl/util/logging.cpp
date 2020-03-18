#include <mbgl/util/logging.hpp>
#include <mbgl/util/enum.hpp>
#include <mbgl/util/platform.hpp>
#include <mbgl/util/thread.hpp>
#include <mbgl/platform/settings.hpp>

#include <cstdio>
#include <cstdarg>
#include <sstream>

namespace mbgl {

namespace {

static std::unique_ptr<Log::Observer> currentObserver;
static bool useThread = true;

} // namespace

class Log::LogThread {
public:
    void record(EventSeverity severity, Event event, int64_t code, const std::string &msg, const optional<std::string>(threadName)) {
        Log::record(severity, event, code, msg, threadName);
    }
};

class Log::Impl {
public:
    Impl() : thread(std::make_unique<util::Thread<LogThread>>(
                    util::makeThreadPrioritySetter(platform::EXPERIMENTAL_THREAD_PRIORITY_WORKER),
                    "LogThread")) {}
        
    void record(EventSeverity severity, Event event, int64_t code, const std::string &msg) {
        if (useThread) {
            thread->actor().invoke(&LogThread::record, severity, event, code, msg, platform::getCurrentThreadName());
        } else {
            Log::record(severity, event, code, msg, {});
        }
    }

private:
    const std::unique_ptr<util::Thread<LogThread>> thread;
};

Log::Log() : impl(std::make_unique<Impl>()) {}

Log* Log::get() noexcept {
    static Log instance;
    return &instance;
}

void Log::useLogThread(bool enable) {
    useThread = enable;
}

void Log::setObserver(std::unique_ptr<Observer> observer) {
    currentObserver = std::move(observer);
}

std::unique_ptr<Log::Observer> Log::removeObserver() {
    std::unique_ptr<Observer> observer;
    std::swap(observer, currentObserver);
    return observer;
}

void Log::record(EventSeverity severity, Event event, const std::string &msg) {
    get()->impl->record(severity, event, -1, msg);
}

void Log::record(EventSeverity severity, Event event, const char* format, ...) {
    va_list args;
    va_start(args, format);
    char msg[4096];
    vsnprintf(msg, sizeof(msg), format, args);
    va_end(args);

    get()->impl->record(severity, event, -1, std::string{ msg });
}

void Log::record(EventSeverity severity, Event event, int64_t code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    char msg[4096];
    vsnprintf(msg, sizeof(msg), format, args);
    va_end(args);

    get()->impl->record(severity, event, code, std::string{ msg });
}

void Log::record(EventSeverity severity, Event event, int64_t code, const std::string &msg, const optional<std::string>& threadName) {
    if (currentObserver && severity != EventSeverity::Debug &&
        currentObserver->onRecord(severity, event, code, msg)) {
        return;
    }

    std::stringstream logStream;

    logStream << "{" << threadName.value_or(platform::getCurrentThreadName()) << "}";
    logStream << "[" << Enum<Event>::toString(event) << "]";

    if (code >= 0) {
        logStream << "(" << code << ")";
    }

    if (!msg.empty()) {
        logStream << ": " << msg;
    }

    platformRecord(severity, logStream.str());
}

} // namespace mbgl

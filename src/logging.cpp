#include "logging.hpp"
#include <cassert>

namespace dtb {

std::unique_ptr<Logger> Logger::instance_;

Logger::Logger(const char* logfile) 
	: enables_{true, true, true}, vlevel_(0)
{
	if(nullptr == logfile)
	{
		ostream_ = make_unique<std::ostream>(std::cerr.rdbuf());
	}
	else
	{
		ostream_ = make_unique<std::ofstream>(logfile);
	}
}

void
Logger::Log(const std::string& msg)
{
	std::lock_guard<std::mutex> lock(log_mtx_);
    *ostream_ << msg << std::endl;
}

void
Logger::Flush()
{
	std::lock_guard<std::mutex> lock(log_mtx_);
	*ostream_ << std::flush;
}

Logger& Logger::instance(const char* logfile)
{
  if ( nullptr == instance_ )
  {
      instance_.reset(new Logger(logfile));
      assert(nullptr != instance_ );
  }
  
  return *instance_;
}

const std::vector<char> LogMessage::severity_name_{'E', 'W', 'I'};

LogMessage::LogMessage(const char* file, int line, Severity severity)
{
  std::string path(file);
  size_t pos = path.rfind('/');
  if (pos != std::string::npos) {
    path = path.substr(pos + 1, std::string::npos);
  }

  stream_ << severity_name_[std::min(severity, Severity::kINFO)] << ' ' << path << ':' << line << "] ";
}

LogMessage::~LogMessage()
{
  Logger::instance().Log(stream_.str());
}

}


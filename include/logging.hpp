#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include "unique.hpp"

namespace dtb {

// A log message.
class LogMessage 
{
 public:
  //The severity corresponding to a log message.
  enum Severity
  {
	 kERROR = 0,		  //An application error has occurred.
	 kWARNING = 1,		  //An application error has been discovered, but TensorRT has recovered or fallen back to a default.
	 kINFO = 2 		      //Informational messages with instructional information.
  };
  
  LogMessage(const char* file, int line, Severity severity);
  ~LogMessage();

  std::stringstream& stream() { return stream_; }

 private:
  static const std::vector<char> severity_name_;
  std::stringstream stream_;
};

// Global logger for messages. Controls how log messages are reported.
class Logger 
{
 public:
  // Is a log level enabled.
  bool IsEnabled(LogMessage::Severity severity) const 
  { 
	return enables_[severity];
  }

  // Set enable for a log Level.
  void SetEnabled(LogMessage::Severity severity, bool enable)
  {
	enables_[severity] = enable;
  }

  // Get the current verbose logging level.
  int VerboseLevel() const { return vlevel_; }

  // Set the current verbose logging level.
  void SetVerboseLevel(int vlevel) { vlevel_ = vlevel; }

  // Log a message.
  void Log(const std::string& msg);

  // Flush the log.
  void Flush();

  static Logger& instance(const char* logfile = nullptr);

 private:
  Logger(const char* logfile);

  static std::unique_ptr<Logger> instance_;
  
  std::vector<bool> enables_;
  int vlevel_;
  std::mutex log_mtx_;
  std::unique_ptr<std::ostream> ostream_;
};

#define LOG_ENABLE_INFO(E)      \
  dtb::Logger::instance().SetEnabled(dtb::LogMessage::Severity::kINFO, (E))
  
#define LOG_ENABLE_WARNING(E)   \
  dtb::Logger::instance().SetEnabled(dtb::LogMessage::Severity::kWARNING, (E))
  
#define LOG_ENABLE_ERROR(E)     \
  dtb::Logger::instance().SetEnabled(dtb::LogMessage::Severity::kERROR, (E))

#define LOG_INFO_IS_ON          \
  dtb::Logger::instance().IsEnabled(dtb::LogMessage::Severity::kINFO)
      
#define LOG_WARNING_IS_ON       \
  dtb::Logger::instance().IsEnabled(dtb::LogMessage::Severity::kWARNING)
  
#define LOG_ERROR_IS_ON         \
  dtb::Logger::instance().IsEnabled(dtb::LogMessage::Severity::kERROR)

#define LOG_INFO                \
  if (LOG_INFO_IS_ON) 			\
	  dtb::LogMessage((char*)__FILE__, __LINE__, dtb::LogMessage::Severity::kINFO).stream()

#define LOG_WARNING             \
  if (LOG_WARNING_IS_ON) 		\
	  dtb::LogMessage((char*)__FILE__, __LINE__, dtb::LogMessage::Severity::kWARNING).stream() 
      
#define LOG_ERROR               \
  if (LOG_ERROR_IS_ON) 			\
	  dtb::LogMessage((char*)__FILE__, __LINE__, dtb::LogMessage::Severity::kERROR).stream()

#define LOG_SET_VERBOSE(L)      \
  dtb::Logger::instance().SetVerboseLevel(static_cast<int>(std::max(0, (L))))

#define LOG_VERBOSE_IS_ON(L)    \
  (dtb::Logger::instance().VerboseLevel() >= (L))

#define LOG_VERBOSE(L)          \
  if (LOG_VERBOSE_IS_ON(L))     \
	  LOG_INFO

#define LOG_FLUSH dtb::Logger::instance().Flush()

}


if("v0.3.4" STREQUAL "")
  message(FATAL_ERROR "Tag for git checkout should not be empty.")
endif()

set(run 0)

if("/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitinfo.txt" IS_NEWER_THAN "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitclone-lastrun.txt")
  set(run 1)
endif()

if(NOT run)
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog'")
endif()

# try the clone 3 times incase there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git" clone "https://github.com/google/glog" "glog"
    WORKING_DIRECTORY "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/glog'")
endif()

execute_process(
  COMMAND "/usr/bin/git" checkout v0.3.4
  WORKING_DIRECTORY "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v0.3.4'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule init
  WORKING_DIRECTORY "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to init submodules in: '/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule update --recursive
  WORKING_DIRECTORY "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitinfo.txt"
    "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitclone-lastrun.txt"
  WORKING_DIRECTORY "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-gitclone-lastrun.txt'")
endif()


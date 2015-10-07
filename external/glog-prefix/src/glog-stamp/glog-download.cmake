

set(command "/usr/local/Cellar/cmake/2.8.12.2/bin/cmake;-P;/Users/kareth/code/studia/piro/caffe/external/glog-prefix/tmp/glog-gitclone.cmake")
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_FILE "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-download-out.log"
  ERROR_FILE "/Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-download-err.log"
  )
if(result)
  set(msg "Command failed: ${result}\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  set(msg "${msg}\nSee also\n  /Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-download-*.log\n")
  message(FATAL_ERROR "${msg}")
else()
  set(msg "glog download command succeeded.  See also /Users/kareth/code/studia/piro/caffe/external/glog-prefix/src/glog-stamp/glog-download-*.log\n")
  message(STATUS "${msg}")
endif()

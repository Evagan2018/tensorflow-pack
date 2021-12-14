/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/system_setup.h"
#include "stdio.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"

extern "C" void stdio_init (void);

namespace tflite {

void debug_log_printf(const char* s) {
		printf("%s", s);
}

// To add an equivalent function for your own platform, create your own
// implementation file, and place it in a subfolder named after the target. See
// tensorflow/lite/micro/debug_log.cc for a similar example.
void InitializeTarget() {
  stdio_init();
	RegisterDebugLogCallback(debug_log_printf);
  //debug_log_printf("Initialized UART and registered Callback.")
}	
}  // namespace tflite

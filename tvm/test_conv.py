from __future__ import absolute_import, print_function
import os

nnvm_dll_path = 'C:\\Users\\pengwa\\Dev\\onnxruntime\\build\\bs\\Windows\\Debug\\external\\tvm_internal\\Debug'
#cl_path = 'C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64'
cl_path = 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86\cl.exe'
os.environ['PATH'] = cl_path + ';' + os.environ['PATH'] + ';' + nnvm_dll_path

import nnvm
import tvm
import numpy as np
import topi

import nnvm
import tvm
import numpy as np
import topi

batch = 1
in_channel = 256
out_channel = 512
in_size = 16
kernel = 3
pad = 1
stride = 1
native_dim = 128

# Algorithm
A = tvm.placeholder((batch, in_size, in_size, in_channel), name='A')
W = tvm.placeholder((kernel, kernel, out_channel, in_channel), name='W')
out_size = (in_size - kernel) // stride + 1
# Create reduction variables
rc = tvm.reduce_axis((0, in_channel), name='rc')
ry = tvm.reduce_axis((0, kernel), name='ry')
rx = tvm.reduce_axis((0, kernel), name='rx')

output_shapes = (batch, out_size, out_size, out_channel)
print(output_shapes)
# Compute the convolution
B = tvm.compute(
    output_shapes,
    lambda nn, yy, xx, ff: tvm.sum(
        W[ry, rx, ff, rc] * A[nn, yy * stride + ry, xx * stride + rx, rc],
        axis=[ry, rx, rc]),
    name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))

'''
(1, 14, 14, 512)
produce B {
  for (yy, 0, 14) {
    for (xx, 0, 14) {
      for (ff, 0, 512) {
        B[((((yy*14) + xx)*512) + ff)] = 0.000000f
        for (ry, 0, 3) {
          for (rx, 0, 3) {
            for (rc, 0, 256) {
              B[((((yy*14) + xx)*512) + ff)] = (B[((((yy*14) + xx)*512) + ff)] + (W[((((ff + (ry*1536)) + (rx*512))*256) + rc)]*A[((((((yy*16) + xx) + (ry*16)) + rx)*256) + rc)]))
            }
          }
        }
      }
    }
  }
}
'''
*/
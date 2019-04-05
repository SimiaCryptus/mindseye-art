/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public interface VisionPipelineLayer {

//  default int[] outputDims(int... inputDims) {
//    return IntStream.range(0, inputDims.length).map(d -> {
//      int inputDim = inputDims[d];
//      if(d < 2) {
//        int inputBorder = getInputBorders()[d];
//        int outputBorder = getOutputBorders()[d];
//        int stride = this.getStrides()[d];
//        return (int) Math.ceil(((double)(inputDim - 2*inputBorder) / stride) + 2*outputBorder);
//      } else if(d == 2) {
//        if (inputDim != getInputChannels()) throw new IllegalArgumentException();
//        return getOutputChannels();
//      } else throw new IllegalArgumentException();
//    }).toArray();
//  }

  String name();

  VisionPipeline<?> getPipeline();

  default PipelineNetwork getNetwork() {
    return ((VisionPipeline<VisionPipelineLayer>) getPipeline()).get(this);
  }

  Layer getLayer();

  int[] getInputBorders();

  int[] getOutputBorders();

  int getInputChannels();

  int getOutputChannels();

  int[] getKernelSize();

  int[] getStrides();
}

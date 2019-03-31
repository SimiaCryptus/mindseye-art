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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.*;

public class VisionPipeline<T extends VisionPipelineLayer> {

  private final LinkedHashMap<T, PipelineNetwork> layers = new LinkedHashMap<>();

  public VisionPipeline(T... values) {
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    for (T value : values) {
      pipelineNetwork.wrap(value.getLayer()).freeRef();
      layers.put(value, (PipelineNetwork) pipelineNetwork.copy().freeze());
    }
    pipelineNetwork.freeRef();
  }

  public Map<T,Tensor> evalFeatureMaps(Tensor input) {
    Map<T,Tensor> tensors = new HashMap<>();
    for (T layer : layers.keySet()) {
      Tensor newValue = layer.getLayer().eval(input).getDataAndFree().getAndFree(0);
      tensors.put(layer,newValue.copy());
      input.freeRef();
      input = newValue;
    }
    return tensors;
  }

  public PipelineNetwork get(T layer) {
    return layers.get(layer);
  }

  public Map<T, PipelineNetwork> getLayers() {
    return Collections.unmodifiableMap(layers);
  }

  ;


}

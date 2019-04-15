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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

public interface VisualModifier {
  PipelineNetwork build(PipelineNetwork network, Tensor... image);

  default PipelineNetwork build(VisionPipelineLayer layer, Tensor... image) {
    PipelineNetwork network = layer.getNetwork();
    network.assertAlive();
    PipelineNetwork pipelineNetwork = build(network, image);
//    network.freeRef();
    return pipelineNetwork;
  }

  ;

  default PipelineNetwork build(Tensor... image) {
    return build((PipelineNetwork) new PipelineNetwork().setName("Input"), image);
  }

  ;

  default VisualModifier combine(VisualModifier right) {
    return (original, image) -> SumInputsLayer.combine(
        this.build(original.copyPipeline(), image),
        right.build(original.copyPipeline(), image)
    );
  }

  default VisualModifier scale(double scale) {
    return (original, image) -> {
      PipelineNetwork build = this.build(original, image);
      build.wrap(new LinearActivationLayer().setScale(scale).freeze());
      return build;
    };
  }

  default VisualModifier pow(double power)
  {
    return (original, image) -> {
      PipelineNetwork build = this.build(original, image);
      build.wrap(new NthPowerActivationLayer().setPower(power).freeze());
      return build;
    };
  }

}

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
    network.freeRef();
    return pipelineNetwork;
  }

  default PipelineNetwork build(Tensor... image) {
    return build((PipelineNetwork) new PipelineNetwork().setName("Input"), image);
  }

  default VisualModifier combine(VisualModifier right) {
    VisualModifier left = this;
    return new VisualModifier() {
      @Override
      public String toString() {
        return String.format("(%s+%s)", left.toString(), right);
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor... image) {
        return (PipelineNetwork) SumInputsLayer.combine(
            VisualModifier.this.build(original, image),
            right.build(original, image)
        ).freeze();
      }
    };
  }

  default VisualModifier scale(double scale) {
    VisualModifier left = this;
    return new VisualModifier() {
      @Override
      public String toString() {
        return String.format("(%s*%s)", left.toString(), scale);
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor... image) {
        PipelineNetwork build = VisualModifier.this.build(original, image);
        build.wrap(new LinearActivationLayer().setScale(scale).freeze());
        return (PipelineNetwork) build.freeze();
      }
    };
  }

  default VisualModifier pow(double power) {
    VisualModifier left = this;
    return new VisualModifier() {
      @Override
      public String toString() {
        return String.format("(%s^%s)", left.toString(), power);
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor... image) {
        PipelineNetwork build = VisualModifier.this.build(original, image);
        build.wrap(new NthPowerActivationLayer().setPower(power).freeze());
        return (PipelineNetwork) build.freeze();
      }
    };
  }

}

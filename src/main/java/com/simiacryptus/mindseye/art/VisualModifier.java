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
import com.simiacryptus.mindseye.util.ImageUtil;
import org.jetbrains.annotations.NotNull;

public interface VisualModifier {
  @NotNull
  public static Tensor resize(Tensor content, Tensor dims) {
    final int[] dimensions = content.getDimensions();
    return Tensor.fromRGB(ImageUtil.resize(dims.toRgbImage(), dimensions[0], dimensions[1]));
  }

  PipelineNetwork build(PipelineNetwork network, Tensor content, Tensor... style);

  default PipelineNetwork build(VisionPipelineLayer layer, Tensor... image) {
    PipelineNetwork network = layer.getNetwork();
    network.assertAlive();
    PipelineNetwork pipelineNetwork = build(network, null, image);
    network.freeRef();
    return pipelineNetwork;
  }

  default PipelineNetwork build(Tensor... image) {
    return build((PipelineNetwork) new PipelineNetwork().setName("Input"), null, image);
  }

  default VisualModifier combine(VisualModifier right) {
    VisualModifier left = this;
    return new VisualModifier() {
      @Override
      public String toString() {
        return String.format("(%s+%s)", left.toString(), right);
      }

      public boolean withMask() {
        return left.withMask() || right.withMask();
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor content, Tensor... style) {
        return (PipelineNetwork) SumInputsLayer.combine(
            VisualModifier.this.build(original, content, style),
            right.build(original, content, style)
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

      public boolean withMask() {
        return left.withMask();
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor content, Tensor... style) {
        PipelineNetwork build = VisualModifier.this.build(original, content, style);
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

      public boolean withMask() {
        return left.withMask();
      }

      @Override
      public PipelineNetwork build(PipelineNetwork original, Tensor content, Tensor... style) {
        PipelineNetwork build = VisualModifier.this.build(original, content, style);
        build.wrap(new NthPowerActivationLayer().setPower(power).freeze());
        return (PipelineNetwork) build.freeze();
      }
    };
  }

  default boolean withMask() {
    return false;
  }

  @NotNull
  default VisualModifier withMask(Tensor maskedInput) {
    final VisualModifier inner = this;
    return new VisualModifier() {
      public boolean withMask() {
        return true;
      }

      @Override
      public PipelineNetwork build(PipelineNetwork network, Tensor content, Tensor... style) {
        final Tensor resizedMaskedInput = resize(content, maskedInput);
        return inner.build(network, resizedMaskedInput, style);
        //return inner.build(gateNetwork(network, finalMask), finalMask, style);
      }
    };

  }
}

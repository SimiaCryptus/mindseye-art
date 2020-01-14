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
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import java.util.function.UnaryOperator;

public interface VisualModifier {

  default boolean isLocalized() {
    return false;
  }

  PipelineNetwork build(VisualModifierParameters visualModifierParameters);

  default PipelineNetwork build(@Nonnull VisionPipelineLayer layer, int[] contentDims, UnaryOperator<Tensor> viewLayer,
                                Tensor... image) {
    PipelineNetwork network = layer.getNetwork();
    network.assertAlive();
    PipelineNetwork pipelineNetwork = build(new VisualModifierParameters(network, contentDims, viewLayer, null, image));
    network.freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  default VisualModifier combine(@Nonnull VisualModifier right) {
    VisualModifier left = this;
    return new VisualModifier() {
      public boolean isLocalized() {
        return left.isLocalized() || right.isLocalized();
      }

      @Nonnull
      @Override
      public String toString() {
        return RefString.format("(%s+%s)", left.toString(), right);
      }

      @Nonnull
      @Override
      public PipelineNetwork build(@Nonnull VisualModifierParameters visualModifierParameters) {
        return (PipelineNetwork) SumInputsLayer.combine(VisualModifier.this.build(visualModifierParameters.addRef()),
            right.build(visualModifierParameters)).freeze();
      }
    };
  }

  @Nonnull
  default VisualModifier scale(double scale) {
    VisualModifier left = this;
    return new VisualModifier() {
      public boolean isLocalized() {
        return left.isLocalized();
      }

      @Nonnull
      @Override
      public String toString() {
        return RefString.format("(%s*%s)", left.toString(), scale);
      }

      @Nonnull
      @Override
      public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
        PipelineNetwork build = VisualModifier.this.build(visualModifierParameters);
        build.add(new LinearActivationLayer().setScale(scale).freeze()).freeRef();
        return (PipelineNetwork) build.freeze();
      }
    };
  }

  @Nonnull
  default VisualModifier pow(double power) {
    VisualModifier left = this;
    return new VisualModifier() {
      public boolean isLocalized() {
        return left.isLocalized();
      }

      @Nonnull
      @Override
      public String toString() {
        return RefString.format("(%s^%s)", left.toString(), power);
      }

      @Nonnull
      @Override
      public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
        PipelineNetwork build = VisualModifier.this.build(visualModifierParameters);
        build.add(new NthPowerActivationLayer().setPower(power).freeze()).freeRef();
        return (PipelineNetwork) build.freeze();
      }
    };
  }

  @Nonnull
  default VisualModifier withMask(Tensor maskedInput) {
    final VisualModifier inner = this;
    return new VisualModifier() {
      public boolean isLocalized() {
        return true;
      }

      @Override
      public PipelineNetwork build(@Nonnull VisualModifierParameters parameters) {
        return inner.build(parameters.withMask(maskedInput));
      }
    };

  }

}

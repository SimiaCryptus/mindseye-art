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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LoggingLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.UnaryOperator;

/**
 * The interface Visual modifier.
 */
public interface VisualModifier {

  /**
   * Is localized boolean.
   *
   * @return the boolean
   */
  default boolean isLocalized() {
    return false;
  }

  /**
   * Build pipeline network.
   *
   * @param visualModifierParameters the visual modifier parameters
   * @return the pipeline network
   */
  PipelineNetwork build(VisualModifierParameters visualModifierParameters);

  /**
   * Build pipeline network.
   *
   * @param layer       the layer
   * @param contentDims the content dims
   * @param viewLayer   the view layer
   * @param image       the image
   * @return the pipeline network
   */
  default PipelineNetwork build(@Nonnull VisionPipelineLayer layer,
                                @Nonnull int[] contentDims,
                                @Nullable UnaryOperator<Tensor> viewLayer,
                                @Nonnull Tensor... image) {
    PipelineNetwork network = layer.getNetwork();
    double scale = layer.getScale();
    layer.freeRef();
    network.assertAlive();
    return build(new VisualModifierParameters(network, scale, contentDims, viewLayer, null, image));
  }

  /**
   * Combine visual modifier.
   *
   * @param right the right
   * @return the visual modifier
   */
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
        Layer layer = SumInputsLayer.combine(
            VisualModifier.this.build(visualModifierParameters.addRef()),
            right.build(visualModifierParameters));
        layer.freeze();
        return (PipelineNetwork) layer;
      }
    };
  }

  /**
   * Scale visual modifier.
   *
   * @param scale the scale
   * @return the visual modifier
   */
  @Nonnull
  default VisualModifier scale(double scale) {
    return RefUtil.wrapInterface(new VisualModifier() {
      public boolean isLocalized() {
        return VisualModifier.this.isLocalized();
      }

      @Nonnull
      @Override
      public String toString() {
        return RefString.format("(%s*%s)", VisualModifier.this.toString(), scale);
      }

      @Nonnull
      @Override
      public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
        PipelineNetwork build = VisualModifier.this.build(visualModifierParameters);
        LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
        linearActivationLayer.setScale(scale);
        linearActivationLayer.freeze();
        build.add(linearActivationLayer).freeRef();
        build.freeze();
        return build;
      }
    }, VisualModifier.this);
  }

  /**
   * With logging visual modifier.
   *
   * @return the visual modifier
   */
  @Nonnull
  default VisualModifier withLogging() {
    return withLogging(getClass().getSimpleName());
  }

  /**
   * With logging visual modifier.
   *
   * @param name the name
   * @return the visual modifier
   */
  @Nonnull
  default VisualModifier withLogging(String name) {
    return new VisualModifier() {
      public boolean isLocalized() {
        return VisualModifier.this.isLocalized();
      }

      @Nonnull
      @Override
      public String toString() {
        return RefString.format("Logging(%s)", VisualModifier.this.toString());
      }

      @Nonnull
      @Override
      public PipelineNetwork build(VisualModifierParameters visualModifierParameters) {
        PipelineNetwork build = VisualModifier.this.build(visualModifierParameters);
        LoggingLayer loggingLayer = new LoggingLayer(LoggingLayer.DetailLevel.Data);
        loggingLayer.setName(name);
        loggingLayer.setLogFeedback(false);
        build.add(loggingLayer).freeRef();
        build.freeze();
        return build;
      }
    };
  }

  /**
   * Pow visual modifier.
   *
   * @param power the power
   * @return the visual modifier
   */
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
        NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
        nthPowerActivationLayer.setPower(power);
        nthPowerActivationLayer.freeze();
        build.add(nthPowerActivationLayer).freeRef();
        build.freeze();
        return build;
      }
    };
  }

  /**
   * With mask visual modifier.
   *
   * @param maskedInput the masked input
   * @return the visual modifier
   */
  @Nonnull
  default VisualModifier withMask(Tensor maskedInput) {
    final VisualModifier inner = this;
    if (maskedInput == null) return inner;
    return RefUtil.wrapInterface(new VisualModifier() {
      public boolean isLocalized() {
        return true;
      }

      @Override
      public PipelineNetwork build(@Nonnull VisualModifierParameters parameters) {
        PipelineNetwork build = inner.build(parameters.withMask(maskedInput.addRef()));
        parameters.freeRef();
        return build;
      }
    }, maskedInput, inner);
  }

}

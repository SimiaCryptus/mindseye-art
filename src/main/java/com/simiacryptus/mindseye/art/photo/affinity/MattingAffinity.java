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

package com.simiacryptus.mindseye.art.photo.affinity;

import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.SimpleRasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import org.ejml.simple.SimpleMatrix;

import javax.annotation.Nonnull;

/**
 * Implements Matting Affinity
 * <p>
 * See Also: A Closed Form Solution to Natural Image Matting
 * http://cs.brown.edu/courses/cs129/results/final/valayshah/Matting-Levin-Lischinski-Weiss-CVPR06.pdf
 * <p>
 */
public class MattingAffinity extends ContextAffinity {
  private double epsilon = 1e-4;

  public MattingAffinity(@Nonnull Tensor content) {
    this(content, new SimpleRasterTopology(content.getDimensions()));
  }

  public MattingAffinity(@Nonnull Tensor content, RasterTopology topology) {
    super(content);
    setTopology(topology);
  }

  public double getEpsilon() {
    return epsilon;
  }

  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MattingAffinity addRef() {
    return (MattingAffinity) super.addRef();
  }

  @Override
  protected double dist(@Nonnull SimpleMatrix vector_i, @Nonnull SimpleMatrix vector_j, @Nonnull SimpleMatrix cov, int neighborhoodSize) {
    int bands = dimensions[2];
    assert neighborhoodSize > 0;
    final SimpleMatrix invert = cov.plus(SimpleMatrix.identity(bands).scale(getEpsilon() / neighborhoodSize)).invert();
    double v = (1.0 + vector_i.dot(invert.mult(vector_j))) / 1;
    v = Math.max(0, v);
    return v;
  }
}

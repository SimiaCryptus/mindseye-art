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

import com.simiacryptus.mindseye.art.photo.MultivariateFrameOfReference;
import com.simiacryptus.mindseye.art.photo.topology.RasterTopology;
import com.simiacryptus.mindseye.art.photo.topology.SimpleRasterTopology;
import com.simiacryptus.mindseye.lang.Tensor;
import org.ejml.simple.SimpleMatrix;

import javax.annotation.Nonnull;

/**
 * Implements experimenal pixel affinity based on logistic model and covariance-normalized distance
 * See Also: https://en.wikipedia.org/wiki/Mahalanobis_distance
 */
public class RelativeAffinity extends ContextAffinity {
  private final double introversion = 8.0;
  private double epsilon = 1e-5;
  private double contrast = 5e0;

  public RelativeAffinity(@Nonnull Tensor content) {
    this(content, new SimpleRasterTopology(content.getDimensions()));
  }

  public RelativeAffinity(@Nonnull Tensor content, RasterTopology topology) {
    super(content);
    setTopology(topology);
  }

  public double getContrast() {
    return contrast;
  }

  public void setContrast(double contrast) {
    this.contrast = contrast;
  }

  public double getEpsilon() {
    return epsilon;
  }

  @Nonnull
  public RelativeAffinity setEpsilon(double epsilon) {
    this.epsilon = epsilon;
    return this;
  }

  public double getIntroversion() {
    return introversion;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RelativeAffinity addRef() {
    return (RelativeAffinity) super.addRef();
  }

  @Override
  protected double dist(@Nonnull SimpleMatrix vector_i, SimpleMatrix vector_j, @Nonnull SimpleMatrix cov, int neighborhoodSize,
                        int globalSize) {
    assert neighborhoodSize > 0;
    final SimpleMatrix invert = MultivariateFrameOfReference.safeInvert(cov, getEpsilon() / neighborhoodSize);
    final SimpleMatrix vect = vector_i.minus(vector_j);
    double v = invert == null ? vect.dot(vect) : vect.dot(invert.mult(vect));
    assert v >= 0;
    v = Math.exp(-getContrast() * v) / getIntroversion();
    return v;
  }
}

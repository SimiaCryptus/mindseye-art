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

package com.simiacryptus.mindseye.art.photo;

import com.simiacryptus.mindseye.lang.Tensor;
import org.ejml.simple.SimpleMatrix;

/**
 * Implements experimenal pixel affinity based on logistic model and covariance-normalized distance
 * See Also: https://en.wikipedia.org/wiki/Mahalanobis_distance
 */
public class RelativeAffinity extends ContextAffinity {
  private final double introversion = 8.0;
  private double epsilon = 1e-8;
  private double contrast = 5e0;

  public RelativeAffinity(Tensor content) {
    this(content, new SimpleRasterTopology(content.getDimensions()));
  }

  public RelativeAffinity(Tensor content, RasterTopology topology) {
    super(content);
    this.setTopology(topology);
  }

  @Override
  protected double dist(SimpleMatrix vector_i, SimpleMatrix vector_j, SimpleMatrix cov, int neighborhoodSize, int globalSize) {
    int bands = dimensions[2];
    assert neighborhoodSize > 0;
    final SimpleMatrix invert = cov.plus(SimpleMatrix.identity(bands).scale(getEpsilon() / neighborhoodSize)).invert();
    final SimpleMatrix vect = vector_i.minus(vector_j);
    double v = vect.dot(invert.mult(vect));
    assert v >= 0;
    v = Math.exp(-getContrast() * v) / getIntroversion();
    return v;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public RelativeAffinity setEpsilon(double epsilon) {
    this.epsilon = epsilon;
    return this;
  }

  public double getContrast() {
    return contrast;
  }

  public RelativeAffinity setContrast(double contrast) {
    this.contrast = contrast;
    return this;
  }

  public double getIntroversion() {
    return introversion;
  }
}


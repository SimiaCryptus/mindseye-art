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

import com.simiacryptus.mindseye.art.photo.affinity.ContextAffinity;
import org.ejml.simple.SimpleMatrix;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Multivariate frame of reference.
 */
public class MultivariateFrameOfReference {
  /**
   * The Inv cov.
   */
  @Nullable
  public final SimpleMatrix invCov;
  /**
   * The Means.
   */
  public final SimpleMatrix means;
  /**
   * The Rms.
   */
  public final SimpleMatrix rms;
  /**
   * The Cov.
   */
  public final SimpleMatrix cov;
  /**
   * The Dimension.
   */
  public final int dimension;

  /**
   * Instantiates a new Multivariate frame of reference.
   *
   * @param a      the a
   * @param b      the b
   * @param mixing the mixing
   */
  public MultivariateFrameOfReference(@Nonnull MultivariateFrameOfReference a, @Nonnull MultivariateFrameOfReference b, double mixing) {
    this(a, b, mixing, 1e-4);
  }

  /**
   * Instantiates a new Multivariate frame of reference.
   *
   * @param a       the a
   * @param b       the b
   * @param mixing  the mixing
   * @param epsilon the epsilon
   */
  public MultivariateFrameOfReference(@Nonnull MultivariateFrameOfReference a, @Nonnull MultivariateFrameOfReference b, double mixing,
                                      double epsilon) {
    means = ContextAffinity.mix(b.means, a.means, mixing);
    rms = ContextAffinity.mix(b.rms, a.rms, mixing);
    cov = ContextAffinity.mix(b.cov, a.cov, mixing);
    this.dimension = a.dimension;
    this.invCov = safeInvert(cov, epsilon);
  }

  /**
   * Instantiates a new Multivariate frame of reference.
   *
   * @param fn       the fn
   * @param channels the channels
   */
  public MultivariateFrameOfReference(@Nonnull Supplier<Stream<double[]>> fn, int channels) {
    this(fn, channels, 1e-4);
  }

  /**
   * Instantiates a new Multivariate frame of reference.
   *
   * @param fn       the fn
   * @param channels the channels
   * @param epsilon  the epsilon
   */
  public MultivariateFrameOfReference(@Nonnull Supplier<Stream<double[]>> fn, int channels, double epsilon) {
    this.dimension = channels;
    means = ContextAffinity.means(fn, this.dimension);
    rms = ContextAffinity.magnitude(means, fn, channels);
    cov = ContextAffinity.covariance(means, rms, fn, channels);
    this.invCov = safeInvert(cov, epsilon);
  }

  /**
   * Safe invert simple matrix.
   *
   * @param cov     the cov
   * @param epsilon the epsilon
   * @return the simple matrix
   */
  @Nullable
  public static SimpleMatrix safeInvert(@Nonnull SimpleMatrix cov, double epsilon) {
    SimpleMatrix invCov;
    try {
      invCov = cov.plus(SimpleMatrix.identity(cov.numCols()).scale(epsilon)).invert();
    } catch (Throwable e) {
      invCov = null;
    }
    return invCov;
  }

  /**
   * Adjust double [ ].
   *
   * @param pixel the pixel
   * @return the double [ ]
   */
  public double[] adjust(@Nonnull double[] pixel) {
    return IntStream.range(0, pixel.length).mapToDouble(c -> (pixel[c] - this.means.get(c)) / this.rms.get(c))
        .toArray();
  }

  /**
   * Dist double.
   *
   * @param vector the vector
   * @return the double
   */
  public double dist(@Nonnull double[] vector) {
    final SimpleMatrix v = ContextAffinity.toMatrix(vector);
    return null == invCov ? v.dot(v) : v.dot(invCov.mult(v));
  }

  /**
   * Raw cov simple matrix.
   *
   * @return the simple matrix
   */
  public SimpleMatrix rawCov() {
    return rms.transpose().mult(cov.mult(rms));
  }

  /**
   * Dist double.
   *
   * @param right the right
   * @return the double
   */
  public double dist(@Nonnull MultivariateFrameOfReference right) {
    final SimpleMatrix sigma0 = this.rawCov();
    final SimpleMatrix sigma1 = right.rawCov();
    final SimpleMatrix sigma_ = new MultivariateFrameOfReference(this, right, 0.5).rawCov();
    final SimpleMatrix mu_diff = this.means.minus(right.means);
    final SimpleMatrix invert = safeInvert(sigma_, 1e-4);
    final double offset = null == invert ? mu_diff.dot(mu_diff) : mu_diff.dot(invert.mult(mu_diff.transpose()));
    final double volume = sigma_.determinant() / Math.sqrt(sigma0.determinant() * sigma1.determinant());
    return offset / 8 + Math.log(volume) / 2;
  }
}

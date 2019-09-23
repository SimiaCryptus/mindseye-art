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

import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class MultivariateFrameOfReference {
  public final SimpleMatrix invCov;
  public final SimpleMatrix means;
  public final SimpleMatrix rms;
  public final SimpleMatrix cov;
  public final int dimension;

  public MultivariateFrameOfReference(MultivariateFrameOfReference a, MultivariateFrameOfReference b, double mixing) {
    this(a, b, mixing, 1e-4);
  }

  public MultivariateFrameOfReference(MultivariateFrameOfReference a, MultivariateFrameOfReference b, double mixing, double epsilon) {
    means = ContextAffinity.mix(b.means, a.means, mixing);
    rms = ContextAffinity.mix(b.rms, a.rms, mixing);
    cov = ContextAffinity.mix(b.cov, a.cov, mixing);
    this.dimension = a.dimension;
    this.invCov = safeInvert(cov, epsilon);
  }

  public MultivariateFrameOfReference(Supplier<Stream<double[]>> fn, int channels) {
    this(fn, channels, 1e-4);
  }

  public MultivariateFrameOfReference(Supplier<Stream<double[]>> fn, int channels, double epsilon) {
    this.dimension = channels;
    means = ContextAffinity.means(fn, this.dimension);
    rms = ContextAffinity.magnitude(means, fn, channels);
    cov = ContextAffinity.covariance(means, rms, fn, channels);
    this.invCov = safeInvert(cov, epsilon);
  }

  public static SimpleMatrix safeInvert(SimpleMatrix cov, double epsilon) {
    SimpleMatrix invCov;
    try {
      invCov = cov.plus(SimpleMatrix.identity(cov.numCols()).scale(epsilon)).invert();
    } catch (Throwable e) {
      invCov = null;
    }
    return invCov;
  }

  public double[] adjust(double[] pixel) {
    return IntStream.range(0, pixel.length).mapToDouble(c -> ((pixel[c]) - this.means.get(c)) / this.rms.get(c)).toArray();
  }

  public double dist(double[] vector) {
    final SimpleMatrix v = ContextAffinity.toMatrix(vector);
    return null == invCov ? v.dot(v) : v.dot(invCov.mult(v));
  }
}

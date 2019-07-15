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

package com.simiacryptus.mindseye.art.util;

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.simiacryptus.mindseye.art.VisionPipelineLayer;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class VisionPipelineUtil {

  private static final Logger log = LoggerFactory.getLogger(VisionPipelineUtil.class);

  public static Closeable cudaReports(NotebookOutput log, boolean interceptLog) {
    Closeable handler_info = log.getHttpd().addGET("cuda/info.txt", "text/plain", outputStream -> {
      try {
        PrintStream stream = new PrintStream(outputStream);
        CudaSystem.printHeader(stream);
        stream.flush();
      } catch (Throwable e) {
        try {
          outputStream.write(Util.toString(e).getBytes("UTF-8"));
        } catch (IOException e1) {
          e1.printStackTrace();
        }
      }
    });
    Closeable handler_stats = log.getHttpd().addGET("cuda/stats.json", "application/json", outputStream -> {
      try {
        PrintStream stream = new PrintStream(outputStream);
        stream.println(JsonUtil.toJson(CudaSystem.getExecutionStatistics()));
        stream.flush();
      } catch (Throwable e) {
        try {
          outputStream.write(Util.toString(e).getBytes("UTF-8"));
        } catch (IOException e1) {
          e1.printStackTrace();
        }
      }
    });
    if (interceptLog) log.subreport("cuda_log", sublog -> {
      CudaSystem.addLog(new Consumer<String>() {
        PrintWriter out;
        long remainingOut = 0;
        long killAt = 0;

        @Override
        public void accept(String formattedMessage) {
          if (null == out) {
            SimpleDateFormat dateFormat = new SimpleDateFormat("dd_HH_mm_ss");
            String date = dateFormat.format(new Date());
            try {
              String caption = String.format("Log at %s", date);
              String filename = String.format("%s_cuda.log", date);
              out = new PrintWriter(sublog.file(filename));
              sublog.p("[%s](etc/%s)", caption, filename);
              sublog.write();
            } catch (Throwable e) {
              throw new RuntimeException(e);
            }
            killAt = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1);
            remainingOut = 10L * 1024 * 1024;
          }
          out.println(formattedMessage);
          out.flush();
          int length = formattedMessage.length();
          remainingOut -= length;
          if (remainingOut < 0 || killAt < System.currentTimeMillis()) {
            out.close();
            out = null;
          }
        }
      });
      return null;
    });

    return new Closeable() {
      @Override
      public void close() throws IOException {
        handler_info.close();
        handler_stats.close();
      }
    };

  }

  @NotNull
  public static Map<String, PipelineNetwork> convertPipeline(GraphDef graphDef, String... nodes) {
    GraphModel graphModel = new GraphModel(graphDef.toByteArray());
    Map<String, PipelineNetwork> graphs = new HashMap<>();
    TFConverter tfConverter = new TFConverter();
    TFLayer tfLayer0 = new TFLayer(
        graphModel.getChild(nodes[0]).subgraph(new HashSet<>(Arrays.asList())).toByteArray(),
        new HashMap<>(),
        nodes[0],
        "input");
    graphs.put(nodes[0], tfConverter.convert(tfLayer0));
    tfLayer0.freeRef();
    for (int i = 1; i < nodes.length; i++) {
      String currentNode = nodes[i];
      String priorNode = nodes[i - 1];
      TFLayer tfLayer1 = new TFLayer(
          graphModel.getChild(currentNode).subgraph(new HashSet<>(Arrays.asList(priorNode))).toByteArray(),
          new HashMap<>(),
          currentNode,
          priorNode);
      graphs.put(currentNode, tfConverter.convert(tfLayer1));
      tfLayer1.freeRef();
    }
    return graphs;
  }

  @NotNull
  public static ArrayList<GraphDef> getNodes(GraphModel graphModel, List<String> nodes) {
    ArrayList<GraphDef> graphs = new ArrayList<>();
    graphs.add(graphModel.getChild(nodes.get(0)).subgraph(new HashSet<>(Arrays.asList())));
    for (int i = 1; i < nodes.size(); i++) {
      graphs.add(graphModel.getChild(nodes.get(i)).subgraph(new HashSet<>(Arrays.asList(nodes.get(i - 1)))));
    }
    return graphs;
  }

  public static void testPinConnectivity(VisionPipelineLayer layer, int... inputDims) {
    DAGNetwork liveTestingNetwork = (DAGNetwork) layer.getLayer();
    liveTestingNetwork.visitLayers(l -> {
      if (l instanceof SimpleConvolutionLayer) {
        Tensor kernel = ((SimpleConvolutionLayer) l).getKernel().map(x -> 1.0);
        ((SimpleConvolutionLayer) l).set(kernel);
        kernel.freeRef();
      } else if (l instanceof ImgBandBiasLayer) {
        ((ImgBandBiasLayer) l).setWeights(x -> 0);
      } else if (l instanceof DAGNetwork) {
        // Ignore
      } else if (!l.state().isEmpty()) {
        throw new RuntimeException(l.getClass().toString());
      }
    });
    int[] outputDims = evalDims(inputDims, liveTestingNetwork.addRef());
    log.info(String.format("testPins(%s,%s) => %s", layer, Arrays.toString(inputDims), Arrays.toString(outputDims)));
    Tensor coordSource = new Tensor(inputDims);
    Map<Coordinate, List<Coordinate>> fwdPinMapping = coordSource.coordStream(true).distinct().filter(x -> x.getCoords()[2] == 0).collect(Collectors.toMap(
        inputPin -> inputPin,
        inputPin -> {
          Tensor testInput = new Tensor(inputDims).setAll(0.0).set(inputPin, 1.0);
          Tensor testOutput = liveTestingNetwork.eval(testInput).getDataAndFree().getAndFree(0).mapAndFree(outValue -> outValue == 0.0 ? 0.0 : 1.0);
          List<Coordinate> coordinates = testOutput.coordStream(true).filter(c -> testOutput.get(c) != 0.0 && c.getCoords()[2] == 0).collect(Collectors.toList());
          testOutput.freeRef();
          testInput.freeRef();
          return coordinates;
        }));
    coordSource.freeRef();
    liveTestingNetwork.freeRef();

    Map<Coordinate, Integer> fwdSizes = fwdPinMapping.entrySet().stream().collect(Collectors.groupingBy(
        e -> e.getKey(), Collectors.summingInt(e -> e.getValue().size())));
    log.info("fwdSizes=" + fwdSizes.entrySet().stream().collect(Collectors.groupingBy(x -> x.getValue(), Collectors.counting())).toString());
    int minDividedKernelSize = IntStream.range(0, 2).map(d -> {
      return (int) Math.floor((double) layer.getKernelSize()[d] / layer.getStrides()[d]);
    }).reduce((a, b) -> a * b).getAsInt();
    int maxDividedKernelSize = IntStream.range(0, 2).map(d -> {
      return (int) Math.ceil((double) layer.getKernelSize()[d] / layer.getStrides()[d]);
    }).reduce((a, b) -> a * b).getAsInt();
    if (!fwdSizes.entrySet().stream().filter(e -> e.getValue() == maxDividedKernelSize).findAny().isPresent()) {
      log.warn("No Fully Represented Input Found");
    }
    int kernelSize = IntStream.range(0, 2).map(d -> {
      return layer.getKernelSize()[d];
    }).reduce((a, b) -> a * b).getAsInt();

    Map<Coordinate, List<Coordinate>> bakPinMapping = fwdPinMapping.entrySet().stream().flatMap(fwdEntry -> fwdEntry.getValue().stream()
        .map(outputCoord -> new Coordinate[]{fwdEntry.getKey(), outputCoord}))
        .collect(Collectors.groupingBy(x -> x[1])).entrySet().stream().collect(Collectors.toMap(
            e -> e.getKey(),
            e -> e.getValue().stream().map(x -> x[0]).collect(Collectors.toList())));
    Map<Coordinate, Integer> bakSizes = bakPinMapping.entrySet().stream().collect(Collectors.groupingBy(
        e -> e.getKey(), Collectors.summingInt(e -> e.getValue().size())));
    log.info("bakSizes=" + bakSizes.entrySet().stream().collect(Collectors.groupingBy(x -> x.getValue(), Collectors.counting())).toString());
    if (!bakSizes.entrySet().stream().filter(e -> e.getValue() == kernelSize).findAny().isPresent()) {
      log.warn("No Fully Represented Output Found");
    }

    fwdSizes.entrySet().stream().filter(e -> e.getValue() > maxDividedKernelSize).forEach(e -> {
      log.info("Overrepresented Input: " + e.getKey() + " = " + e.getValue());
    });
    fwdSizes.entrySet().stream().filter(e -> e.getValue() < minDividedKernelSize).forEach(e -> {
      int[] coords = e.getKey().getCoords();
      int[] inputBorders = layer.getInputBorders();
      int[] array = IntStream.range(0, inputBorders.length).filter(d -> {
        if (inputBorders[d] > coords[d]) return true;
        return ((inputDims[d]) - inputBorders[d]) <= coords[d];
      }).toArray();
      if (array.length == 0) {
        log.warn("Underrepresented Input: " + e.getKey() + " = " + e.getValue());
      }
    });

    bakSizes.entrySet().stream().filter(e -> e.getValue() < kernelSize).forEach(e -> {
      int[] coords = e.getKey().getCoords();
      int[] outputBorders = layer.getOutputBorders();
      int[] array = IntStream.range(0, outputBorders.length).filter(d -> {
        if (outputBorders[d] > coords[d]) return true;
        return ((outputDims[d]) - outputBorders[d]) <= coords[d];
      }).toArray();
      if (0 == array.length) {
        log.warn("Underrepresented Output: " + e.getKey() + " = " + e.getValue());
      }
    });
  }

  public static int[] evalDims(int[] inputDims, Layer layer) {
    Tensor input = new Tensor(inputDims);
    Tensor tensor = layer.eval(input).getDataAndFree().getAndFree(0);
    input.freeRef();
    int[] dimensions = tensor.getDimensions();
    tensor.freeRef();
    layer.freeRef();
    return dimensions;
  }

  @Nonnull
  public static BufferedImage load(final CharSequence image, final int imageSize) {
    BufferedImage source = getImage(image);
    return imageSize <= 0 ? source : TestUtil.resize(source, imageSize, true);
  }

  @Nonnull
  public static BufferedImage load(final CharSequence image, final int width, final int height) {
    BufferedImage source = getImage(image);
    return width <= 0 ? source : TestUtil.resize(source, width, height);
  }

  @Nonnull
  public static BufferedImage getImage(final CharSequence file) {
    if (file.toString().startsWith("http")) {
      try {
        BufferedImage read = ImageIO.read(new URL(file.toString()));
        if (null == read) throw new IllegalArgumentException("Error reading " + file);
        return read;
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    }
    FileSystem fileSystem = getFileSystem(file.toString());
    Path path = new Path(file.toString());
    try {
      if (!fileSystem.exists(path)) throw new IllegalArgumentException("Not Found: " + path);
      try (FSDataInputStream open = fileSystem.open(path)) {
        byte[] bytes = IOUtils.toByteArray(open);
        try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) {
          return ImageIO.read(in);
        }
      }
    } catch (Throwable e) {
      throw new RuntimeException("Error reading " + file, e);
    }
  }

  public static FileSystem getFileSystem(final CharSequence file) {
    Configuration conf = getHadoopConfig();
    FileSystem fileSystem;
    try {
      fileSystem = FileSystem.get(new Path(file.toString()).toUri(), conf);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return fileSystem;
  }

  @Nonnull
  public static Configuration getHadoopConfig() {
    Configuration configuration = new Configuration(false);

    File tempDir = new File("temp");
    tempDir.mkdirs();
    configuration.set("hadoop.tmp.dir", tempDir.getAbsolutePath());
//    configuration.set("fs.http.impl", org.apache.hadoop.fs.http.HttpFileSystem.class.getCanonicalName());
//    configuration.set("fs.https.impl", org.apache.hadoop.fs.http.HttpsFileSystem.class.getCanonicalName());
    configuration.set("fs.git.impl", com.simiacryptus.hadoop_jgit.GitFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.aws.credentials.provider", DefaultAWSCredentialsProviderChain.class.getCanonicalName());
    return configuration;
  }

  public static int[][] getIndexMap(final SimpleConvolutionLayer layer) {
    int[] kernelDimensions = layer.getKernelDimensions();
    double b = Math.sqrt(kernelDimensions[2]);
    int h = kernelDimensions[1];
    int w = kernelDimensions[0];
    int l = (int) (w * h * b);
    return IntStream.range(0, (int) b).mapToObj(i -> {
      return IntStream.range(0, l).map(j -> j + l * i).toArray();
    }).toArray(i -> new int[i][]);
  }
}

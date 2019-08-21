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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.tensorflow.TFLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.UploadImageQuery;
import com.simiacryptus.tensorflow.GraphModel;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.jetbrains.annotations.NotNull;
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
import java.util.stream.IntStream;

public class ImageArtUtil {

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
    if (interceptLog) log.subreport(sublog -> {
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
    }, log.getName() + "_" + "cuda_log");

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

  public static BufferedImage load(NotebookOutput log, final CharSequence image, final int imageSize) {
    return ImageUtil.load(() -> getImage(image, log), imageSize);
  }

  public static BufferedImage load(NotebookOutput log, final CharSequence image, final int width, final int height) {
    return ImageUtil.load(() -> getImage(image, log), width, height);
  }

  public static Tensor loadTensor(NotebookOutput log, final CharSequence image, final int imageSize) {
    return getImageTensor(image, log, imageSize, imageSize);
  }

  public static Tensor loadTensor(NotebookOutput log, final CharSequence image, final int width, final int height) {
    return getImageTensor(image, log, width, height);
  }

  public static BufferedImage getImage(@Nonnull final CharSequence file, NotebookOutput log) {
    Tensor imageTensor = getImageTensor(file, log);
    return null==imageTensor?null:imageTensor.toImage();
  }

  public static Tensor getImageTensor(@Nonnull final CharSequence file, NotebookOutput log) {
    String fileStr = file.toString();
    int length = fileStr.split("\\:")[0].length();
    if (length <= 0 || length >= Math.min(7,fileStr.length())) {
      if (fileStr.contains(" * ")) {
        Tensor sampleImage = Arrays.stream(fileStr.split(" * ")).map(x -> getImageTensor(x, log)).filter(x -> x != null).findFirst().get();
        return Arrays.stream(fileStr.split(" * ")).map(x -> getImageTensor(x, log, sampleImage.getDimensions()[0], sampleImage.getDimensions()[1])).reduce((a, b) -> {
          Tensor r = a.mapCoordsAndFree(c -> a.get(c) * b.get(c));
          b.freeRef();
          return r;
        }).get();
      } else if (fileStr.contains(" + ")) {
        Tensor sampleImage = Arrays.stream(fileStr.split(" + ")).map(x -> getImageTensor(x, log)).filter(x -> x != null).findFirst().get();
        return Arrays.stream(fileStr.split(" + ")).map(x -> getImageTensor(x, log, sampleImage.getDimensions()[0], sampleImage.getDimensions()[1])).reduce((a, b) -> {
          Tensor r = a.mapCoordsAndFree(c -> a.get(c) + b.get(c));
          b.freeRef();
          return r;
        }).get();
      }
      return null;
    }
    return getTensor(log, file);
  }

  @Nonnull
  public static Tensor getImageTensor(@Nonnull final CharSequence file, NotebookOutput log, int width, int height) {
    String fileStr = file.toString();
    int length = fileStr.split("\\:")[0].length();
    if (length <= 0 || length >= Math.min(7,fileStr.length())) {
      if (fileStr.contains(" + ")) {
        Tensor sampleImage = Arrays.stream(fileStr.split(" +\\+ +")).map(x -> getImageTensor(x, log, width, height)).filter(x -> x != null).findFirst().get();
        return Arrays.stream(fileStr.split(" +\\+ +")).map(x -> getImageTensor(x, log, sampleImage.getDimensions()[0], sampleImage.getDimensions()[1])).reduce((a, b) -> {
          Tensor r = a.mapCoordsAndFree(c -> a.get(c) + b.get(c));
          b.freeRef();
          return r;
        }).get();
      } else if (fileStr.contains(" * ")) {
        Tensor sampleImage = Arrays.stream(fileStr.split(" +\\* +")).map(x -> getImageTensor(x, log, width, height)).filter(x -> x != null).findFirst().get();
        return Arrays.stream(fileStr.split(" +\\* +")).map(x -> getImageTensor(x, log, sampleImage.getDimensions()[0], sampleImage.getDimensions()[1])).reduce((a, b) -> {
          Tensor r = a.mapCoordsAndFree(c -> a.get(c) * b.get(c));
          b.freeRef();
          return r;
        }).get();
      } else if (fileStr.trim().toLowerCase().equals("plasma")) {
        return new Plasma().paint(width, height);
      } else if (fileStr.trim().toLowerCase().equals("noise")) {
        return new Tensor(width, height, 3).mapAndFree(x -> FastRandom.INSTANCE.random() * 100);
      } else if (fileStr.matches("\\-?\\d+(?:\\.\\d*)?(?:[eE]\\-?\\d+)?")) {
        double v = Double.parseDouble(fileStr);
        return new Tensor(width, height, 3).mapAndFree(x -> v);
      }
    }
    return getTensor(log, file);
  }

  @NotNull
  public static Tensor getTensor(NotebookOutput log, @Nonnull CharSequence file) {
    String fileStr = file.toString();
    if (fileStr.trim().toLowerCase().startsWith("upload:")) {
      String key = fileStr.substring("upload:".length());
      MarkdownNotebookOutput markdownLog = (MarkdownNotebookOutput) log;
      return (Tensor) markdownLog.uploadCache.computeIfAbsent(key, k -> {
        try {
          UploadImageQuery uploadImageQuery = new UploadImageQuery(k, log);
          return Tensor.fromRGB(ImageIO.read(uploadImageQuery.print().get()));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    } else if (fileStr.startsWith("http")) {
      try {
        BufferedImage read = ImageIO.read(new URL(fileStr));
        if (null == read) throw new IllegalArgumentException("Error reading " + file);
        return Tensor.fromRGB(read);
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    }
    FileSystem fileSystem = getFileSystem(fileStr);
    Path path = new Path(fileStr);
    try {
      if (!fileSystem.exists(path)) throw new IllegalArgumentException("Not Found: " + path);
      try (FSDataInputStream open = fileSystem.open(path)) {
        byte[] bytes = IOUtils.toByteArray(open);
        try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) {
          return Tensor.fromRGB(ImageIO.read(in));
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

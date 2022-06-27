# TFDetection
TFDetection is a detection toolbox based on Tensorflow2.


## Overview of Components
<div align = "center">
  <a href = "https://github.com/Burf/TFDetection/blob/main/tfdet/model/detector/__init__.py"><b>Detector</b></a>
</div>
<table align = "center">
  <tbody>
    <tr align = "center" valign = "bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Segmentation</b>
      </td>
      <td>
        <b>Anomaly Detection</b>
      </td>
    </tr>
    <tr valign = "top">
      <td>
        <ul>
          <li>Faster R-CNN(2015)</li>
          <li>RetinaNet(2017)</li>
          <li>YoloV3(2018)</li>
          <li>YoloV3 Tiny(2018)</li>
          <li>Cascade R-CNN(2018)</li>
          <li>FCOS(2019)</li>
          <li>Hybrid Task Cascade(2019)</li>
          <li>EfficientDet(2019)</li>
          <li>YoloV4(2020)</li>
          <li>YoloV4 Tiny(2020)</li>
          <li>EfficientDet Lite(2020)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>Semantic Segmentation</b></li>
          <ul>
            <li>FCN(2015)</li>
            <li>UNet(2016)</li>
            <li>PSPNet(2017)</li>
            <li>DeepLab V3(2017)</li>
            <li>DeepLab V3+(2018)</li>
            <li>UNet++(2018)</li>
            <li>UperNet(2018)</li>
          </ul>
        </ul>
        <ul>
          <li><b>Instance Segmentation</b></li>
          <ul>
            <li>Mask R-CNN(2017)</li>
            <li>Cascade R-CNN(2018)</li>
            <li>Hybrid Task Cascade(2019)</li>
          </ul>
        </ul>
      </td>
      <td>
        <ul>
          <li>SPADE(2020)</li>
          <li>PaDiM(2020)</li>
          <li>PatchCore(2021)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
 
<div align = "center">
  <b>Components</b>
</div>
<table align = "center">
  <tbody>
    <tr align = "center" valign = "bottom">
      <td>
        <a href = "https://github.com/Burf/TFDetection/blob/main/tfdet/model/backbone/__init__.py"><b>Backbone</b></a>
      </td>
      <td>
        <a href = "https://github.com/Burf/TFDetection/blob/main/tfdet/model/neck/__init__.py"><b>Neck</b></a>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign = "top">
      <td>
        <ul>
          <li>VGGNet(2015)</li>
          <li>ResNet(2016)</li>
          <li>Wide ResNet(2016)</li>
          <li>DenseNet(2016)</li>
          <li>ResNeXt(2016)</li>
          <li>MobileNet(2017)</li>
          <li>MobileNetV2(2018)</li>
          <li>DarkNet(2018)</li>
          <li>MobileNetV3(2019)</li>
          <li>EfficientNet(2019)</li>
          <li>ResNeSt(2020)</li>
          <li>CSP DarkNet(2020)</li>
          <li>SwinTransformer(2021)</li>
          <li>EfficientNetV2(2021)</li>
          <li>SwinTransformerV2(2022)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>FPN(2017)</li>
          <li>PANet(2018)</li>
          <li>BiFPN(2019)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href = "https://github.com/Burf/TFDetection/blob/main/tfdet/core/assign/__init__.py"><b>Assign</b></a></li>
          <ul>
            <li>Max Iou(2015)</li>
            <li>Center Region(2019)</li>
            <li>Point(2019)</li>
            <li>Adaptive Training Sample Selection(2020)</li>
          </ul>
        </ul>
        <ul>
          <li><a href = "https://github.com/Burf/TFDetection/blob/main/tfdet/util/metric.py"><b>Metric</b></a></li>
          <ul>
            <li>Mean Average Precision</li>
          </ul>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


## Requirements

- Python 3
- tensorflow 2.4▲ (If you installed tensorflow 2.8▲, more backbone is available.)
- keras
- opencv-python


## Contributor

 * Hyungjin Kim(flslzk@gmail.com)

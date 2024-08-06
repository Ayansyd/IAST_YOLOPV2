import argparse
import time
from pathlib import Path
import cv2
import torch
import requests
import json

from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, show_seg_result, \
    AverageMeter, LoadImages

class_names = [
    'person', 'bicycle', 'airplane', 'car', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def plot_one_box(xyxy, img, label=None, color=None, line_thickness=3):
    if color is None:
        color = [0, 255, 0]
    x1, y1, x2, y2 = map(int, xyxy)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
    
    if label:
        font_scale = 0.5
        font_thickness = line_thickness - 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
        img = cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)
        img = cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, [0, 0, 0], font_thickness, lineType=cv2.LINE_AA)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--tomtom-api-key', type=str, required=True, help='TomTom API key')
    parser.add_argument('--start', type=str, required=True, help='Start coordinates (latitude,longitude)')
    parser.add_argument('--end', type=str, required=True, help='End coordinates (latitude,longitude)')
    return parser

def get_route(api_key, start, end):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json?key={api_key}&routeType=fastest&traffic=false"
    response = requests.get(url)
    if response.status_code == 200:
        route_data = response.json()
        if validate_route(route_data):
            print(route_data)
            return route_data
        else:
            print("Invalid route data received.")
            return None
    else:
        print(f"Failed to get route: {response.status_code}")
        return None

def validate_route(route_data):
    try:
        routes = route_data['routes']
        if len(routes) == 0:
            print("No routes found.")
            return False
        legs = routes[0]['legs']
        if len(legs) == 0:
            print("No legs found in the route.")
            return False
        points = legs[0]['points']
        if len(points) == 0:
            print("No points found in the leg.")
            return False
        return True
    except KeyError as e:
        print(f"Missing key in route data: {e}")
        return False

def plot_route_on_image(img, route):
    for leg in route['routes'][0]['legs']:
        for point in leg['points']:
            lat = point['latitude']
            lon = point['longitude']
            # Convert lat/lon to image coordinates (this requires a proper mapping, which is not included in this example)
            # Here, just simulate with random points for illustration
            x, y = int(lon * 10) % img.shape[1], int(lat * 10) % img.shape[0]
            img = cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img

def detect(opt):
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    stride = 32
    device = select_device(opt.device)

    try:
        model = torch.jit.load(weights, map_location=device)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("The model may be CUDA-only or saved with CUDA-specific optimizations. Please check the model file and ensure it is compatible with the selected device.")
        return

    half = device.type != 'cpu'
    model = model.to(device)

    if half:
        model.half()
    model.eval()

    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    route = get_route(opt.tomtom_api_key, opt.start, opt.end)
    if route:
        print("Route obtained successfully")

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        inf_time.update(t2 - t1, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))

        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in det:
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label)

            print(f"{s}Done. ({t2 - t1:.3f}s)")

            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            if route:
                im0 = plot_route_on_image(im0, route)

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fourcc = 'mp4v'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # Display the video in real-time
            cv2.imshow('Video', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break

    print('Results saved to %s' % save_dir)
    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = make_parser()
    opt = parser.parse_args()
    detect(opt)
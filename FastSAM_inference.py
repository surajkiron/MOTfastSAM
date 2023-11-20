import torch
from PIL import Image
import sys
sys.path.insert(0,"/home/skn/arpl/Projects/MOTfastSAM/FastSAM")
from fastsam import FastSAM, FastSAMPrompt 
from utils.tools import convert_box_xywh_to_xyxy
h= 640
w = 480

class get_fSAM():

    def __init__(self):
        # FastSAM Arguments
        self.model_path = "FastSAM/weights/FastSAM.pt"
        self.output = "resources/output/"
        self.imgsz = 800

        self.iou = 0.9 # iou threshold for filtering the annotations
        self.text_prompt = None#"human"
        self.conf = 0.4 # object confidence threshold
        self.randomcolor = True
        self.point_prompt = [[0,0]]
        self.point_label = [0]
        self.box_prompt = [[0,0,0,0]]
        self.better_quality = False
        self.retina = True
        self.withContours = False

        self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        print("device: ", self.device)


    def infer(self, img_path):
        # load model
        model = FastSAM(self.model_path)
        point_prompt = self.point_prompt
        box_prompt = convert_box_xywh_to_xyxy(self.box_prompt)
        point_label = self.point_label
        input = Image.open(img_path)
        input = input.convert("RGB")
        everything_results = model(
            input,
            device=self.device,
            retina_masks=self.retina,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou    
            )
        bboxes = None
        points = None
        point_label = None
        prompt_process = FastSAMPrompt(input, everything_results, device=self.device)
        if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
                print("Box Prompt is ", self.box_prompt_prompt)
                ann = prompt_process.box_prompt(bboxes=box_prompt)
                bboxes = box_prompt
        elif self.text_prompt != None:
            print("Text Prompt is ", self.text_prompt)
            ann = prompt_process.text_prompt(text=self.text_prompt)
        elif point_prompt[0] != [0, 0]:
            print("Point Prompt is ", self.point_prompt)
            ann = prompt_process.point_prompt(
                points=point_prompt, pointlabel=point_label
            )
            points = point_prompt
            point_label = point_label
        else:
            ann = prompt_process.everything_prompt()
            output = prompt_process.plot(
            annotations=ann,
            output_path=self.output+img_path.split("/")[-1],
            bboxes = bboxes,
            points = points,
            point_label = point_label,
            withContours=self.withContours,
            better_quality=self.better_quality,)

        return output




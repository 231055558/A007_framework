from dataset.transform.basetransform import BaseTransform


class CenterCrop(BaseTransform):
    def __init__(self, crop_size):
        assert isinstance(crop_size, tuple) and len(crop_size) == 2
        self.crop_size = crop_size

    def transform(self, results):
        img = results['img']
        h, w, c = img.shape
        crop_h, crop_w = self.crop_size

        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        cropped_img = img[start_h:end_h, start_w:end_w, :]
        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape
        return results

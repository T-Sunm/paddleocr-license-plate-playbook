import paddle
from ppocr.modeling.architectures.distillation_model import DistillationModel


class DistillationModelLRLP(DistillationModel):
    def __init__(self, config):
        super().__init__(config)
        self._student_idx = None
        for idx, name in enumerate(self.model_name_list):
            if name == "Student":
                self._student_idx = idx
                break

    def forward(self, x, data=None):
        # Eval mode: only run Student, but return dict with both keys for PostProcess compatibility
        if not self.training:
            student_out = self.model_list[self._student_idx](x, data)
            return {"Student": student_out, "Teacher": student_out}

        # Train mode: run both Teacher and Student
        result_dict = dict()
        x_lr = x

        if data is not None and len(data) > 0:
            x_hr = data[-1]
            data_student = data[:-1]
        else:
            x_hr = x
            data_student = data

        for idx, model_name in enumerate(self.model_name_list):
            model = self.model_list[idx]
            if model_name == "Teacher":
                was_training = model.training
                model.train()
                with paddle.no_grad():
                    result_dict[model_name] = model(x_hr, data_student)
                if not was_training:
                    model.eval()
            else:
                result_dict[model_name] = model(x_lr, data_student)

        return result_dict

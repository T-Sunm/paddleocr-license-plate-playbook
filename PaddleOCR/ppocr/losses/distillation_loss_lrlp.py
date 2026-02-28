import paddle
from ppocr.losses.distillation_loss import DistillationDMLLoss, _sum_loss
from ppocr.losses.basic_loss import DMLLoss


class DistillationNRTRDMLLossLRLP(DistillationDMLLoss):
    """Fixed version that calls DMLLoss.forward() directly."""

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]

            if self.multi_head:
                loss = DMLLoss.forward(self, out1[self.dis_head], out2[self.dis_head])
            else:
                loss = DMLLoss.forward(self, out1, out2)
            
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, idx)] = loss

        loss_dict = _sum_loss(loss_dict)
        return loss_dict

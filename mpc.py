import torch

# 直接指定设备为 CPU
device = torch.device("cpu")

class mpc_member:
    def __init__(self, model_path, id, num_random = 5):
        # 直接将模型的 state_dict 赋值给 model_param
        # 修改此处，将 weights_only 设置为 False
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model_param = checkpoint.get('state_dict', {}).copy()

        # 初始化 random_model_param
        self.random_model_param = {}
        for param_name, param_value in self.model_param.items():
            shape = param_value.shape + (num_random,)
            self.random_model_param[param_name] = torch.zeros(shape, device=device)

        # 初始化 random_vectors
        self.random_vectors = {}
        for param_name, param_value in self.model_param.items():
            shape = param_value.shape + (num_random,)
            self.random_vectors[param_name] = torch.zeros(shape, device=device)

        # 初始化 id
        self.id = id
        self.num_random = num_random

    def initialize_random_vectors(self):
        # 初始化 random_vectors（秘密分享方式）
        for param_name, param_value in self.model_param.items():
            shape = param_value.shape
            # 生成前 num_random - 1 个随机值
            random_tensor = (torch.rand(*shape, self.num_random - 1, device=device) * 2 - 1) * (0.1 / self.num_random)
            # 第 num_random 个值为补偿值，使得 sum = 原始值
            last_value = param_value.unsqueeze(-1) - random_tensor.sum(dim=-1, keepdim=True)
            self.random_vectors[param_name] = torch.cat([random_tensor, last_value], dim=-1)

        # random_model_param 保持不变
        for param_name, param_value in self.model_param.items():
            shape = param_value.shape
            random_tensor = (torch.rand(*shape, self.num_random - 1, device=device) * 2 - 1) * (0.1 / self.num_random)
            last_value = param_value.unsqueeze(-1) - random_tensor.sum(dim=-1, keepdim=True)
            self.random_model_param[param_name] = torch.cat([random_tensor, last_value], dim=-1)

    def set_id(self, new_id):
        self.id = new_id

    def get_random_vectors_with_zeroed_id(self):
        # 复制 self.random_vectors
        random_vectors_copy = {k: v.clone() for k, v in self.random_vectors.items()}
        # 将第 self.id 个元素设为 0
        for param_name in random_vectors_copy.keys():
            random_vectors_copy[param_name][..., self.id] = torch.zeros_like(random_vectors_copy[param_name][..., self.id])
        return random_vectors_copy

    def calculate_sum_with_lobby_dic(self, lobby_dic):
        result = {}
        for param_name in self.random_vectors.keys():
            result[param_name] = self.random_vectors[param_name][..., self.id] + lobby_dic[param_name][..., self.id]
        return result

class mpc_lobby:
    def __init__(self, num_member):
        self.num_member = num_member

    def aggregate_random_vectors(self, members):
        if len(members) != self.num_member:
            raise ValueError(f"Input member number should be {self.num_member}")

        result = None
        for member in members:
            vectors = member.get_random_vectors_with_zeroed_id()
            if result is None:
                result = vectors.copy()
            else:
                for param_name in vectors.keys():
                    result[param_name] += vectors[param_name]
        return result

    def calculate_average(self, member_dicts):
        if len(member_dicts) != self.num_member:
            raise ValueError(f"Input member number should be {self.num_member}")

        sum_dict = {}
        for member_dict in member_dicts:
            for param_name, param_value in member_dict.items():
                if param_name not in sum_dict:
                    sum_dict[param_name] = param_value.clone()
                else:
                    sum_dict[param_name] += param_value

        average_dict = {param_name: param_value / self.num_member for param_name, param_value in sum_dict.items()}
        return average_dict


if __name__ == "__main__":
    model_paths = ['./raw_data/averaged_model1.pth',
                   './raw_data/averaged_model2.pth',
                   './raw_data/averaged_model3.pth',
                   './raw_data/averaged_model4.pth',
                   './raw_data/averaged_model5.pth']
    
    members = []
    num_random = 5
    for id, model_path in enumerate(model_paths):
        members.append(mpc_member(model_path, id, num_random))
    for member in members:
        member.initialize_random_vectors()
    lobby = mpc_lobby(len(model_paths))
    lobby_dic = lobby.aggregate_random_vectors(members)

    member_dicts = []
    for member in members:
        member_dicts.append(member.calculate_sum_with_lobby_dic(lobby_dic))

    average_dict = lobby.calculate_average(member_dicts)

    # 计算 manual average，不再重新加载模型，只用 members 中的 model_param
    manual_avg = {}
    for key in members[0].model_param:
        stacked = torch.stack([m.model_param[key] for m in members])
        manual_avg[key] = stacked.mean(dim=0)

    # 对比 manual_avg 和 average_dict
    mismatch_found = False
    for key in manual_avg:
        if not torch.allclose(manual_avg[key], average_dict[key], atol=1e-5, rtol=1e-3):
            print(f"[验证失败] 参数 {key} 不一致")
            mismatch_found = True
    if not mismatch_found:
        print("✅ average_dict 确实是5个模型的算术平均值！")

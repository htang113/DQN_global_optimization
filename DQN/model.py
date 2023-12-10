import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn

class Q_net(torch.nn.Module):
    
    def __init__(self, device, min_radius: float = 0.1, max_radius: float = 2, emb_neurons: int = 8) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.max_radius = max_radius;
        self.min_radius = min_radius;
        self.device = device;

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.irreps_input = o3.Irreps("1o");
        irreps_mid1 = o3.Irreps("8x0e + 4x1o + 4x2e");
        irreps_mid2 = o3.Irreps("8x0e + 4x1o + 4x2e");
        irreps_output = o3.Irreps("0e");

        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_input,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid1,
            shared_weights=False
        )
        self.tp2 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid1,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid2,
            shared_weights=False
        )
        self.tp3 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_output,
            shared_weights=False
        )
        
        self.fc1 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp1.weight_numel], torch.relu);
        self.fc2 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp2.weight_numel], torch.relu);
        self.fc3 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp3.weight_numel], torch.relu);

    def forward(self, pos, actions) -> torch.Tensor:
        
        nframe = len(pos);
        natm = len(pos[0]);
        num_nodes = nframe*natm;
        pos = pos.reshape([-1,3]);
        f_in = actions.reshape([-1,3]);

        batch = torch.tensor([int(i//natm) for i in range(num_nodes)]).to(self.device);
        
        edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);
        self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
        edge_src = torch.cat((edge_src, self_edge));
        edge_dst = torch.cat((edge_dst, self_edge));
        
        edge_vec = pos[edge_src] - pos[edge_dst];
        num_neighbors = len(edge_src) / num_nodes;

        sh = o3.spherical_harmonics(l = self.irreps_sh, 
                                    x = edge_vec, 
                                    normalize=True, 
                                    normalization='component').to(self.device)
        
        rnorm = edge_vec.norm(dim=1);
        crit1, crit2 = rnorm<self.max_radius, rnorm>self.min_radius;
        emb = (torch.cos(rnorm/self.max_radius*torch.pi)+1)/2; 
        emb = (emb*crit1*crit2 + (~crit2)).reshape(len(edge_src),1);
        
        edge_feature = self.tp1(f_in[edge_src], sh, self.fc1(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        edge_feature = self.tp2(node_feature[edge_src], sh, self.fc2(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        edge_feature = self.tp3(node_feature[edge_src], sh, self.fc3(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        Vlist = node_feature.reshape([nframe, natm])
        Vlist = torch.sum(Vlist,axis=1);
        
        return Vlist;


# =============================================================================
# class value_net(torch.nn.Module):
#     
#     def __init__(self, device, min_radius: float = 0.5, max_radius: float = 2.5, emb_neurons: int = 16) -> None:
#         super().__init__()
#         
#         # Initialize a Equivariance graph convolutional neural network
#         # nodes with distance smaller than max_radius are connected by bonds
#         # num_basis is the number of basis for edge feature embedding
#         
#         self.max_radius = max_radius;
#         self.min_radius = min_radius;
#         self.device = device;
# 
#         self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
#         self.irreps_input = o3.Irreps("0e");
#         irreps_mid1 = o3.Irreps("8x0e + 8x1o + 8x2e");
#         irreps_mid2 = o3.Irreps("8x0e + 8x1o + 8x2e");
#         irreps_output = o3.Irreps("0e");
# 
#         self.tp1 = o3.FullyConnectedTensorProduct(
#             irreps_in1=self.irreps_input,
#             irreps_in2=self.irreps_sh,
#             irreps_out=irreps_mid1,
#             shared_weights=False
#         )
#         self.tp2 = o3.FullyConnectedTensorProduct(
#             irreps_in1=irreps_mid1,
#             irreps_in2=self.irreps_sh,
#             irreps_out=irreps_mid2,
#             shared_weights=False
#         )
#         self.tp3 = o3.FullyConnectedTensorProduct(
#             irreps_in1=irreps_mid2,
#             irreps_in2=self.irreps_sh,
#             irreps_out=irreps_output,
#             shared_weights=False
#         )
#         
#         self.fc1 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp1.weight_numel], torch.relu);
#         self.fc2 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp2.weight_numel], torch.relu);
#         self.fc3 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp3.weight_numel], torch.relu);
# 
#     def forward(self, pos) -> torch.Tensor:
#         
#         nframe = len(pos);
#         natm = len(pos[0]);
#         num_nodes = nframe*natm;
#         pos = pos.reshape([-1,3]);
#         f_in = torch.ones([num_nodes,1]).to(self.device);
# 
#         batch = torch.tensor([int(i//natm) for i in range(num_nodes)]).to(self.device);
#         
#         edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);
#         self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
#         edge_src = torch.cat((edge_src, self_edge));
#         edge_dst = torch.cat((edge_dst, self_edge));
#         
#         edge_vec = pos[edge_src] - pos[edge_dst];
#         num_neighbors = len(edge_src) / num_nodes;
# 
#         sh = o3.spherical_harmonics(l = self.irreps_sh, 
#                                     x = edge_vec, 
#                                     normalize=True, 
#                                     normalization='component').to(self.device)
#         
#         rnorm = edge_vec.norm(dim=1);
#         crit1, crit2 = rnorm<self.max_radius, rnorm>self.min_radius;
#         emb = (torch.cos(rnorm/self.max_radius*torch.pi)+1)/2; 
#         emb = (emb*crit1*crit2 + (~crit2)).reshape(len(edge_src),1);
#         
#         edge_feature = self.tp1(f_in[edge_src], sh, self.fc1(emb));
#         node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
#         edge_feature = self.tp2(node_feature[edge_src], sh, self.fc2(emb));
#         node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
#         edge_feature = self.tp3(node_feature[edge_src], sh, self.fc3(emb));
#         node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
#         Vlist = node_feature.reshape([nframe, natm])
#         Vlist = torch.sum(Vlist,axis=1);
#         
#         return Vlist;
# 
# device = 'cuda:0';
# policy = policy_net(device);
# policy.to(device);
# value = value_net(device);
# value.to(device);
# pos = torch.randn([100,38,3]).to(device);
# actions = torch.randn([100,38,3]).to(device);
# 
# P = policy(pos,actions)
# V = value(pos)
# =============================================================================

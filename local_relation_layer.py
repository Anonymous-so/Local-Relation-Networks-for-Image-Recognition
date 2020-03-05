import torch


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def forward(self):
        # as the paper does not infer how to construct a 2xkxk position matrix
        # we assume that it's a kxk matrix for deltax,and a kxk matric for deltay.
        # that is, [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]] for kernel = 3
        a_range  = torch.arange(-1*(self.k//2),(self.k//2)+1).view(1,-1)
        x_position = a_range.expand(self.k,a_range.shape[1])
        b_range = torch.arange((self.k//2),-1*(self.k//2)-1,-1).view(-1,1)
        y_position = b_range.expand(b_range.shape[0],self.k)
        position = torch.cat((x_position.unsqueeze(0),y_position.unsqueeze(0)),0).unsqueeze(0).float()
        if torch.cuda.is_available():
            position = position.cuda()
        out = self.l2(torch.nn.functional.relu(self.l1(position)))
        return out



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)
    
    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map).transpose(2,1).contiguous()   # [N batch , H_out*Wout, C channel * k*k]
        query_map_unfold = self.unfold(query_map).transpose(2,1).contiguous()    # [N batch , H_out*Wout, C channel * k*k]
        key_map_unfold = key_map_unfold.view(key_map.shape[0],-1, key_map.shape[1], key_map_unfold.shape[-1]//key_map.shape[1])
        query_map_unfold = query_map_unfold.view(query_map.shape[0], -1, query_map.shape[1], query_map_unfold.shape[-1]//query_map.shape[1])
        key_map_unfold = key_map_unfold.transpose(2,1).contiguous()
        query_map_unfold = query_map_unfold.transpose(2,1).contiguous()
        return (key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]).view(key_map_unfold.shape[0],key_map_unfold.shape[1],key_map_unfold.shape[2],k,k)    #[N batch, C channel, (H-k+1)*(W-k+1), k*k]


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel,dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k, stride=1, padding =0,m=None):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, self.padding, self.stride)
        self.gp = GeometryPrior(k, channels//m)
        self.unfold = torch.nn.Unfold(k, 1, self.padding, self.stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):   # x = [N,C,H,W]
        km = self.kmap(x)       # [N,C/m,h,w]
        qm = self.qmap(x)       # [N,C/m,h,w]
        ak = self.ac((km, qm))  # [N,C/m,H_out*W_out, k,k]
        gpk = self.gp()    # [1, C/m,k,k]
        ck = combine_prior(ak, gpk.unsqueeze(2))[:, None, :, :, :]  # [N,1,C/m,H_out*W_out, k,k]
        x_unfold = self.unfold(x).transpose(2,1).contiguous().view(x.shape[0], -1, x.shape[1], self.k*self.k).transpose(2,1).contiguous()
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m, -1, self.k, self.k)     # [N, m, C/m, H_out*W_out, k,k]
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1], -1, self.k*self.k)     #  [N, C,HOUT*WOUT, k*k]
        h_out = (x.shape[2] + 2 * self.padding - 1 * self.k )//  self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * self.k )//  self.stride + 1
        pre_output = torch.sum(pre_output, 3).view(x.shape[0], x.shape[1], h_out, w_out)    # [N, C, H_out*W_out]
        return self.final1x1(pre_output)

if __name__ == '__main__':
    # example of local relation layer
    layer = LocalRelationalLayer(channels=64,k=7,stride=1,padding=0, m=8).cuda()
    input = torch.zeros(2,64,19,19).cuda()
    output = layer(input)

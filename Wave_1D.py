import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x)
        return x

model = NeuralNet()

def loss(model, x, t, c):
    f = model(x, t)
    der_f_t = torch.autograd.grad(f, t, create_graph=True, grad_outputs=torch.ones_like(f))[0]
    der_f_tt = torch.autograd.grad(der_f_t, t, create_graph=True, grad_outputs=torch.ones_like(der_f_t))[0]

    der_f_x = torch.autograd.grad(f, x, create_graph=True, grad_outputs=torch.ones_like(f))[0]
    der_f_xx = torch.autograd.grad(der_f_x, x, create_graph=True, grad_outputs=torch.ones_like(der_f_x))[0]

    return ((der_f_tt - (c**2) * der_f_xx) ** 2).sum()

def loss_IC(model, x, t, sigma):
    f = model(x, t)
    initial = torch.exp(-((x - 7.5)**2) / (2 * sigma**2))

    return ((f - initial) ** 2).sum()
    
def loss_bound(model, x, t):
    f = model(x, t)
    return (f ** 2).sum()

def vel_loss(model, x, t):
    f = model(x, t)
    der_f_t = torch.autograd.grad(f, t, create_graph=True, grad_outputs=torch.ones_like(f))[0]

    return (der_f_t ** 2).sum()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50,T_mult=2)
L = 15
T = 20
num_points = 6000
epochs = 30000

x_pde = torch.rand(num_points, 1) * L
t_pde = torch.rand(num_points, 1) * T
x_pde.requires_grad = True
t_pde.requires_grad = True

x_ic = torch.rand(num_points, 1) * L
t_ic = torch.zeros(num_points, 1)
x_ic.requires_grad = True
t_ic.requires_grad = True


t_bc = torch.rand(num_points, 1) * T
x_bc0 = torch.zeros(num_points, 1)
x_bcL = torch.ones(num_points, 1) * L  


x_bc0.requires_grad = True
t_bc.requires_grad = True
x_bcL.requires_grad = True

for epoch in range(epochs):

    if epoch%1000 == 0:

        x_pde = torch.rand(num_points, 1) * L
        t_pde = torch.rand(num_points, 1) * T
        x_pde.requires_grad = True
        t_pde.requires_grad = True

        x_ic = torch.rand(num_points, 1) * L
        t_ic = torch.zeros(num_points, 1)
        x_ic.requires_grad = True
        t_ic.requires_grad = True   

        t_bc = torch.rand(num_points, 1) * T
        x_bc0 = torch.zeros(num_points, 1)
        x_bcL = torch.ones(num_points, 1) * L  


        x_bc0.requires_grad = True
        t_bc.requires_grad = True
        x_bcL.requires_grad = True

    optimizer.zero_grad()

    loss_pde = loss(model, x_pde, t_pde, c=1)
    loss_ic_val = loss_IC(model, x_ic, t_ic, sigma=0.5)
    loss_b1 = loss_bound(model, x_bc0, t_bc)
    loss_b2 = loss_bound(model, x_bcL, t_bc)
    loss_vel = vel_loss(model, x_ic, t_ic) 

    cost = 100*loss_pde + loss_ic_val +  loss_vel + 5 * (loss_b1 + loss_b2) 
    cost.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    scheduler.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {cost.item()}')

step = 10000
x = torch.linspace(0, L, step, device=device).unsqueeze(1)
t_values = torch.linspace(0, 50, 1000, device=device).unsqueeze(1)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("U(x, t)")
ax.set_title("Wave Over Time")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    t = t_values[frame].expand_as(x)
    with torch.no_grad():
        y = model(x, t).cpu().numpy()
    line.set_data(x.cpu().numpy(), y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_values), init_func=init, blit=True)
ani.save('Result.gif',writer=animation.PillowWriter(fps=25))
plt.show()

torch.save(model.state_dict(), 'checkpoint.pth')
import numpy as np

class VoxelMatrix:
    def __init__(self, nx, ny, nz):
        """Inicializa a matriz 3D de zeros usando NumPy para alta performance."""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.matriz = np.zeros((nx, ny, nz), dtype=int)
        
    def export_powermeis(self, filename, compactado=True):
        """Unifica a exportação para o formato .mtx lido pelo PowerMEIS."""
        if compactado:
            with open(filename, 'w') as f:
                print(self.nx, self.ny, self.nz, '0', file=f)
                
                iCp_old = self.matriz[0, 0, self.nz-1]
                c = 0
                
                for k in range(self.nz-1, -1, -1):
                    for j in range(self.ny):
                        for i in range(self.nx):
                            iCp = self.matriz[i, j, k]
                            if (iCp != iCp_old) or (c == 50000):
                                print(c, iCp_old, file=f)
                                c = 0
                                iCp_old = iCp
                            c += 1
                print(c, iCp_old, file=f)
        else:
            with open(filename, 'w') as f:
                for k in range(self.nz-1, -1, -1):
                    for j in range(self.ny):
                        for i in range(self.nx):
                            print(self.matriz[i, j, k], end=' ', file=f)
                        print('', file=f)
                    print('', file=f)

class PrismaTriangular(VoxelMatrix):
    def __init__(self, base, altura, ri=1):
        super().__init__(nx=base, ny=base, nz=altura)
        
        for i in range(self.nx):
            for j in range(self.ny):
                if j <= i and (i + j) <= (base - 1):
                    self.matriz[i, j, :] = ri

class CuboCoreShell(VoxelMatrix):
    def __init__(self, lado_total, lado_core, altura, ri1=1, ri2=2):
        super().__init__(nx=lado_total, ny=lado_total, nz=altura)
        
        x0, y0, z0 = self.nx // 2, self.ny // 2, self.nz // 2
        half_total = lado_total // 2
        half_core = lado_core // 2
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if (abs(i - x0) <= half_total and abs(j - y0) <= half_total and abs(k - z0) <= half_total):
                        if (abs(i - x0) <= half_core and abs(j - y0) <= half_core and abs(k - z0) <= half_core):
                            self.matriz[i, j, k] = ri1
                        else:
                            self.matriz[i, j, k] = ri2

class CilindroEnterrado(VoxelMatrix):
    def __init__(self, nx_dim, ny_dim, h_cilindro, h_total, raio, ri=1, hi=2):
        # A dimensão original do script soma 1 em x/y e 2 em z
        super().__init__(nx=nx_dim + 1, ny=ny_dim + 1, nz=h_total + 2)
        
        x0 = (nx_dim + 1) // 2
        y0 = (ny_dim + 1) // 2
        
        # Otimização NumPy para varredura do cilindro
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    d2 = (i - x0)**2 + (j - y0)**2
                    if d2 <= raio**2:
                        self.matriz[i, j, k] = ri
                    elif d2 > raio**2 and k > h_cilindro:
                        self.matriz[i, j, k] = hi

class EsferaCoreShell(VoxelMatrix):
    def __init__(self, r_shell, r_core, i1=1, i2=2):
        nx = int(2 * r_shell)
        ny = int(2 * r_shell)
        nz = int(2 * r_shell)
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        centro = (nx - 1) / 2.0
        
        # Otimização extrema com vetorização do NumPy (sem loops for lentos)
        I, J, K = np.ogrid[:nx, :ny, :nz]
        dist_sq = (I - centro)**2 + (J - centro)**2 + (K - centro)**2
        
        # Preenche casca (i1) e depois sobrescreve o núcleo (i2)
        self.matriz[dist_sq <= r_shell**2] = i1
        self.matriz[dist_sq <= r_core**2] = i2

class EsferoideSubstrato(VoxelMatrix):
    def __init__(self, a, b, c, h_subs, ri=1, hi=2, half=False):
        nx = 2 * a + 1
        ny = 2 * b + 1
        nz = 2 * c + 1
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        x0, y0, z0 = nx // 2, ny // 2, nz // 2
        nz0 = z0 if half else 0
        
        inv_a2 = 1 / (a**2)
        inv_b2 = 1 / (b**2)
        inv_c2 = 1 / (c**2)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(nz0, self.nz):
                    d2 = inv_a2*(i - x0)**2 + inv_b2*(j - y0)**2 + inv_c2*(k - z0)**2
                    if d2 <= 1:
                        self.matriz[i, j, k] = ri
                    elif k < h_subs:
                        self.matriz[i, j, k] = hi

class Cubo(VoxelMatrix):
    def __init__(self, nx, ny, nz, cx, cy, cz, z_offset=0, ri=1):
        # Cria a "caixa" total preenchida com zeros (vácuo)
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        # Centraliza o cubo no eixo X e Y da matriz maior
        x_start = (nx - cx) // 2
        y_start = (ny - cy) // 2
        
        # Preenche apenas a região do cubo com o índice do material.
        # O resto da matriz continuará sendo zero (vácuo).
        self.matriz[x_start : x_start + cx, 
                    y_start : y_start + cy, 
                    z_offset : z_offset + cz] = ri


class CuboMulticamada(VoxelMatrix):
    def __init__(self, nx, ny, nz, cx, cy, cz_base, cz_filme, z_offset=0, ri_base=1, ri_filme=2):
        # Caixa total de vácuo
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        # Centraliza a estrutura nos eixos X e Y
        x_start = (nx - cx) // 2
        y_start = (ny - cy) // 2
        
        # 1. Preenche a base do cubo
        self.matriz[x_start : x_start + cx, 
                    y_start : y_start + cy, 
                    z_offset : z_offset + cz_base] = ri_base
        
        # 2. Preenche o filme exatamente em cima da base
        z_filme_start = z_offset + cz_base
        self.matriz[x_start : x_start + cx, 
                    y_start : y_start + cy, 
                    z_filme_start : z_filme_start + cz_filme] = ri_filme


class StackBuilderMatrix(VoxelMatrix):
    def __init__(self, nx, ny, layers):
        # A altura total (nz) é a soma das espessuras de todas as camadas
        nz_total = sum(layer['z'] for layer in layers)
        super().__init__(nx=nx, ny=ny, nz=nz_total)
        
        current_z = 0
        for layer in layers:
            dz = layer['z']
            ri = layer['ri']
            # Preenche o plano X,Y inteiro com a espessura dz atual
            self.matriz[:, :, current_z : current_z + dz] = ri
            current_z += dz

class ImportedMatrix(VoxelMatrix):
    def __init__(self, matriz_numpy_3d):
        # O parser vai entregar uma matriz no formato do NumPy/IRVM (nz, ny, nx).
        # Precisamos extrair as dimensões e converter para o formato do nosso construtor (nx, ny, nz).
        nz, ny, nx = matriz_numpy_3d.shape
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        # Transpõe os eixos para o padrão PowerMEIS Builder (X, Y, Z)
        self.matriz = np.transpose(matriz_numpy_3d, (2, 1, 0))

class AdvancedStackMatrix(VoxelMatrix):
    def __init__(self, nx, ny, layers):
        # A altura total (nz) continua sendo a soma das espessuras de todas as camadas
        nz_total = sum(layer['cz'] for layer in layers)
        super().__init__(nx=nx, ny=ny, nz=nz_total)
        
        current_z = 0
        for layer in layers:
            cx = layer['cx']
            cy = layer['cy']
            cz = layer['cz']
            ri = layer['ri']
            align = layer['align']
            
            # Calcula as coordenadas de início (X, Y) baseadas no alinhamento escolhido
            if align == 'Center':
                start_x = (nx - cx) // 2
                start_y = (ny - cy) // 2
            elif align == 'Top-Left':
                start_x, start_y = 0, 0
            elif align == 'Top-Right':
                start_x, start_y = nx - cx, 0
            elif align == 'Bottom-Left':
                start_x, start_y = 0, ny - cy
            elif align == 'Bottom-Right':
                start_x, start_y = nx - cx, ny - cy
            else:  # Custom Offset
                start_x = layer.get('offset_x', 0)
                start_y = layer.get('offset_y', 0)
                
            # Prevenção de erro: se a camada for empurrada para fora, cortamos o excesso
            end_x = min(start_x + cx, nx)
            end_y = min(start_y + cy, ny)
            start_x = max(start_x, 0)
            start_y = max(start_y, 0)
            
            # Pinta apenas o bloco específico dentro desse "andar" (Z)
            self.matriz[start_x:end_x, start_y:end_y, current_z : current_z + cz] = ri
            
            current_z += cz


class MembranaPorosa(VoxelMatrix):
    def __init__(self, nx, ny, nz, esp, ri, poros_config, seed=42):
        # Inicia a matriz de vácuo
        super().__init__(nx=nx, ny=ny, nz=nz)
        
        # Preenche o material base até a espessura 'esp'
        self.matriz[:, :, :esp] = ri
        
        cz = esp // 2  # Centro dos poros no eixo Z
        if seed is not None:
            np.random.seed(seed)
            
        # poros_config será uma lista de tuplas ex: [(25, 10), (50, 4)] -> [(Raio, Quantidade)]
        for R, Q in poros_config:
            for _ in range(Q):
                # Sorteia o centro X e Y evitando que a esfera corte a borda da matriz
                # Fallback de segurança caso o raio seja maior que a matriz
                cx = np.random.randint(R, nx - R) if nx > 2*R else nx // 2
                cy = np.random.randint(R, ny - R) if ny > 2*R else ny // 2
                
                # Define os limites da caixa delimitadora (bounding box)
                x_min, x_max = max(0, cx - R), min(nx, cx + R + 1)
                y_min, y_max = max(0, cy - R), min(ny, cy + R + 1)
                z_min, z_max = max(0, cz - R), min(nz, cz + R + 1)
                
                # Cria as coordenadas locais dentro da caixa
                x, y, z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
                
                # Equação da esfera
                mascara_esfera = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= R**2
                
                # Aplica a máscara na sub-região da matriz (esculpindo o poro com 0 / vácuo)
                self.matriz[x_min:x_max, y_min:y_max, z_min:z_max][mascara_esfera] = 0


class IRVMMatrix(VoxelMatrix):
    def __init__(self, mask_base, mask_stack, nz_base, nz_stack, ri_base=1, ri_stack=2):
        # As máscaras (imagens) entram no formato (ny, nx)
        ny, nx = mask_base.shape
        nz_total = nz_base + nz_stack
        
        # Inicializa a matriz 3D preenchida com zeros
        super().__init__(nx=nx, ny=ny, nz=nz_total)
        
        # Transpõe as máscaras de (ny, nx) para (nx, ny) para alinhar com o nosso eixo (X, Y, Z)
        mb_t = mask_base.T
        ms_t = mask_stack.T
        
        # Vetorização mágica do NumPy (substitui os loops lentos):
        # 1. Encontra onde existe base OU stack
        base_area = np.logical_or(mb_t == 1, ms_t == 1)
        
        # 2. Preenche a espessura da base (de z=0 até nz_base) nesses locais
        self.matriz[base_area, :nz_base] = ri_base
        
        # 3. Preenche a espessura do stack (de nz_base até o topo) apenas onde tem stack
        self.matriz[ms_t == 1, nz_base:] = ri_stack
from geometrias import PrismaTriangular, CuboCoreShell

def testar_matrizes():
    print("Iniciando o Construtor de Matrizes PowerMEIS...")

    # 1. Testando o Prisma Triangular
    print("\n[1] Gerando Prisma Triangular...")
    prisma = PrismaTriangular(base=50, altura=100, ri=1)
    prisma.export_powermeis("teste_prisma_compactado.mtx", compactado=True)
    print("-> Arquivo 'teste_prisma_compactado.mtx' salvo com sucesso!")

    # 2. Testando o Cubo Core/Shell
    print("\n[2] Gerando Cubo Core/Shell...")
    cubo = CuboCoreShell(lado_total=60, lado_core=30, altura=60, ri1=2, ri2=1)
    cubo.export_powermeis("teste_cubo_coreshell.mtx", compactado=True)
    print("-> Arquivo 'teste_cubo_coreshell.mtx' salvo com sucesso!")

if __name__ == "__main__":
    testar_matrizes()
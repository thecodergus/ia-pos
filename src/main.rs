use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

use std::f64::consts::PI;

use std::fs::File;
use std::io::Write;

// Constantes
const DIMENSIONS: usize = 5; // Altere para o número de dimensões desejado
const GLOBAL_BEST: f64 = 0.0; // Melhor valor global da função de custo
const B_LO: f64 = -5.0; // Limite inferior do espaço de busca
const B_HI: f64 = 5.0; // Limite superior do espaço de busca

const POPULATION: usize = 30; // Número de partículas no enxame
const V_MAX: f64 = 0.1; // Velocidade máxima
const CONVERGENCE: f64 = 1e-50; // Critério de convergência
const MAX_ITER: usize = 100_000; // Número máximo de iterações

// Estrutura da Partícula
#[derive(Clone)]
struct Particle {
    pos: [f64; DIMENSIONS],
    pos_z: f64,
    velocity: [f64; DIMENSIONS],
    best_pos: [f64; DIMENSIONS],
    best_pos_z: f64,
}

// Estrutura do Enxame
struct Swarm {
    particles: Vec<Particle>,
    best_pos: [f64; DIMENSIONS],
    best_pos_z: f64,
}

// Estrutura para armazenar informações de cada iteração
struct Iteration {
    iteration_number: usize,
    best_value: f64,
    average_value: f64,
}

// Implementação das funções para o Enxame
impl Swarm {
    fn new(population: usize, v_max: f64) -> Swarm {
        let mut particles = Vec::new();
        let mut best_pos = [0.0; DIMENSIONS];
        let mut best_pos_z = std::f64::INFINITY;
        let mut rng = rand::thread_rng();

        for _ in 0..population {
            let mut pos = [0.0; DIMENSIONS];
            let mut velocity = [0.0; DIMENSIONS];
            for i in 0..DIMENSIONS {
                pos[i] = rng.gen_range(B_LO..B_HI);
                velocity[i] = rng.gen_range(-v_max..v_max);
            }
            let pos_z = cost_function(&pos);
            let particle = Particle {
                pos,
                pos_z,
                velocity,
                best_pos: pos,
                best_pos_z: pos_z,
            };
            if particle.pos_z < best_pos_z {
                best_pos = particle.pos;
                best_pos_z = particle.pos_z;
            }
            particles.push(particle);
        }

        Swarm {
            particles,
            best_pos,
            best_pos_z,
        }
    }
}

// Função de custo (Ackley) generalizada para N dimensões
fn cost_function(pos: &[f64; DIMENSIONS]) -> f64 {
    let a: f64 = 20.0;
    let b: f64 = 0.2;
    let c: f64 = 2.0 * PI;

    let sum_sq: f64 = pos.iter().map(|&xi| xi.powi(2)).sum();
    let sum_cos: f64 = pos.iter().map(|&xi| (c * xi).cos()).sum();

    let term_1: f64 = (-b * (sum_sq / DIMENSIONS as f64).sqrt()).exp();
    let term_2: f64 = (sum_cos / DIMENSIONS as f64).exp();

    -a * term_1 - term_2 + a + std::f64::consts::E
}

// Versão Base do PSO
fn particle_swarm_optimization_base() -> Vec<Iteration> {
    // Inicializa o enxame
    let mut swarm: Swarm = Swarm::new(POPULATION, V_MAX);

    let mut curr_iter: usize = 0;

    // Vetor para armazenar as informações de cada iteração
    let mut iterations_data: Vec<Iteration> = Vec::new();

    while curr_iter < MAX_ITER {
        // Paraleliza a atualização das partículas
        let particles_clone: &mut Vec<Particle> = &mut swarm.particles;
        let best_particle = particles_clone
            .par_iter_mut()
            .map(|particle| {
                let mut rng = rand::thread_rng();

                for i in 0..DIMENSIONS {
                    let r1: f64 = rng.gen_range(0.0..1.0);
                    let r2: f64 = rng.gen_range(0.0..1.0);

                    // Coeficientes
                    let c1 = 2.0;
                    let c2 = 2.0;

                    // Atualiza a velocidade da partícula
                    let personal_coefficient = c1 * r1 * (particle.best_pos[i] - particle.pos[i]);
                    let social_coefficient = c2 * r2 * (swarm.best_pos[i] - particle.pos[i]);
                    let mut new_velocity =
                        particle.velocity[i] + personal_coefficient + social_coefficient;

                    // Verifica se a velocidade excede o máximo
                    if new_velocity > V_MAX {
                        new_velocity = V_MAX;
                    } else if new_velocity < -V_MAX {
                        new_velocity = -V_MAX;
                    }
                    particle.velocity[i] = new_velocity;
                }

                // Atualiza a posição atual da partícula
                for i in 0..DIMENSIONS {
                    particle.pos[i] += particle.velocity[i];

                    // Verifica se a partícula está dentro dos limites
                    if particle.pos[i] > B_HI || particle.pos[i] < B_LO {
                        particle.pos[i] = rng.gen_range(B_LO..B_HI);
                    }
                }
                particle.pos_z = cost_function(&particle.pos);

                // Atualiza a melhor posição conhecida da partícula
                if particle.pos_z < particle.best_pos_z {
                    particle.best_pos = particle.pos;
                    particle.best_pos_z = particle.pos_z;
                }

                // Retorna a partícula para possível atualização do melhor global
                particle.clone()
            })
            // Encontra a melhor partícula desta iteração
            .min_by(|a, b| a.pos_z.partial_cmp(&b.pos_z).unwrap())
            .unwrap();

        // Atualiza a melhor posição global se necessário
        if best_particle.pos_z < swarm.best_pos_z {
            swarm.best_pos = best_particle.pos;
            swarm.best_pos_z = best_particle.pos_z;
        }

        // Calcula a média dos valores de pos_z das partículas nesta iteração
        let sum_pos_z: f64 = swarm.particles.par_iter().map(|p| p.pos_z).sum();
        let average_pos_z: f64 = sum_pos_z / POPULATION as f64;

        // Armazena os dados desta iteração
        iterations_data.push(Iteration {
            iteration_number: curr_iter,
            best_value: swarm.best_pos_z,
            average_value: average_pos_z,
        });

        // Verifica a convergência
        if (swarm.best_pos_z - GLOBAL_BEST).abs() < CONVERGENCE {
            println!(
                "O enxame atingiu o critério de convergência após {} iterações (Base).",
                curr_iter
            );
            break;
        }
        curr_iter += 1;
    }

    println!("Melhor posição encontrada (Base): {:?}", swarm.best_pos);
    println!("Melhor valor encontrado (Base): {}", swarm.best_pos_z);

    // Salva o melhor vetor de posição em um arquivo
    let mut file: File =
        File::create("best_position_base.txt").expect("Não foi possível criar o arquivo");
    writeln!(file, "{:?}", swarm.best_pos).expect("Não foi possível escrever no arquivo");

    // Retorna o vetor com as informações de cada iteração
    iterations_data
}

// Função PSO com Peso de Inércia (w)
fn particle_swarm_optimization_w(w: f64) -> Vec<Iteration> {
    // Inicializa o enxame
    let mut swarm: Swarm = Swarm::new(POPULATION, V_MAX);

    let mut curr_iter: usize = 0;

    // Vetor para armazenar as informações de cada iteração
    let mut iterations_data = Vec::new();

    while curr_iter < MAX_ITER {
        // Paraleliza a atualização das partículas
        let particles_clone = &mut swarm.particles;
        let best_particle = particles_clone
            .par_iter_mut()
            .map(|particle| {
                let mut rng = rand::thread_rng();

                for i in 0..DIMENSIONS {
                    let r1: f64 = rng.gen_range(0.0..1.0);
                    let r2: f64 = rng.gen_range(0.0..1.0);

                    // Coeficientes
                    let c1: f64 = 2.0;
                    let c2: f64 = 2.0;

                    // Atualiza a velocidade da partícula
                    let personal_coefficient = c1 * r1 * (particle.best_pos[i] - particle.pos[i]);
                    let social_coefficient = c2 * r2 * (swarm.best_pos[i] - particle.pos[i]);
                    let mut new_velocity =
                        w * particle.velocity[i] + personal_coefficient + social_coefficient;

                    // Verifica se a velocidade excede o máximo
                    if new_velocity > V_MAX {
                        new_velocity = V_MAX;
                    } else if new_velocity < -V_MAX {
                        new_velocity = -V_MAX;
                    }
                    particle.velocity[i] = new_velocity;
                }

                // Atualiza a posição atual da partícula
                for i in 0..DIMENSIONS {
                    particle.pos[i] += particle.velocity[i];

                    // Verifica se a partícula está dentro dos limites
                    if particle.pos[i] > B_HI || particle.pos[i] < B_LO {
                        particle.pos[i] = rng.gen_range(B_LO..B_HI);
                    }
                }
                particle.pos_z = cost_function(&particle.pos);

                // Atualiza a melhor posição conhecida da partícula
                if particle.pos_z < particle.best_pos_z {
                    particle.best_pos = particle.pos;
                    particle.best_pos_z = particle.pos_z;
                }

                // Retorna a partícula para possível atualização do melhor global
                particle.clone()
            })
            // Encontra a melhor partícula desta iteração
            .min_by(|a, b| a.pos_z.partial_cmp(&b.pos_z).unwrap())
            .unwrap();

        // Atualiza a melhor posição global se necessário
        if best_particle.pos_z < swarm.best_pos_z {
            swarm.best_pos = best_particle.pos;
            swarm.best_pos_z = best_particle.pos_z;
        }

        // Calcula a média dos valores de pos_z das partículas nesta iteração
        let sum_pos_z: f64 = swarm.particles.par_iter().map(|p| p.pos_z).sum();
        let average_pos_z: f64 = sum_pos_z / POPULATION as f64;

        // Armazena os dados desta iteração
        iterations_data.push(Iteration {
            iteration_number: curr_iter,
            best_value: swarm.best_pos_z,
            average_value: average_pos_z,
        });

        // Verifica a convergência
        if (swarm.best_pos_z - GLOBAL_BEST).abs() < CONVERGENCE {
            println!(
                "O enxame atingiu o critério de convergência após {} iterações (w).",
                curr_iter
            );
            break;
        }
        curr_iter += 1;
    }

    println!("Melhor posição encontrada (w): {:?}", swarm.best_pos);
    println!("Melhor valor encontrado (w): {}", swarm.best_pos_z);

    // Salva o melhor vetor de posição em um arquivo
    let mut file = File::create("best_position_w.txt").expect("Não foi possível criar o arquivo");
    writeln!(file, "{:?}", swarm.best_pos).expect("Não foi possível escrever no arquivo");

    // Retorna o vetor com as informações de cada iteração
    iterations_data
}

// Função PSO com Fator de Constrição (k)
fn particle_swarm_optimization_k() -> Vec<Iteration> {
    // Inicializa o enxame
    let mut swarm: Swarm = Swarm::new(POPULATION, V_MAX);

    let mut rng: ThreadRng = rand::thread_rng();

    // Definir c1 e c2 de forma que φ > 4
    let c1: f64 = 2.05;
    let c2: f64 = 2.05;
    let phi: f64 = c1 + c2;

    if phi <= 4.0 {
        panic!("Phi deve ser maior que 4 para o cálculo do fator de constrição.");
    }

    let sqrt_term: f64 = ((phi * phi) - (4.0 * phi)).sqrt();
    let denominator: f64 = (2.0 - phi - sqrt_term).abs();
    let k: f64 = 2.0 / denominator;

    let mut curr_iter: usize = 0;

    // Vetor para armazenar as informações de cada iteração
    let mut iterations_data: Vec<Iteration> = Vec::new();

    while curr_iter < MAX_ITER {
        // Paraleliza a atualização das partículas
        let particles_clone = &mut swarm.particles;
        let best_particle = particles_clone
            .par_iter_mut()
            .map(|particle| {
                let mut rng = rand::thread_rng();

                for i in 0..DIMENSIONS {
                    let r1: f64 = rng.gen_range(0.0..1.0);
                    let r2: f64 = rng.gen_range(0.0..1.0);

                    // Atualiza a velocidade da partícula
                    let cognitive_component = c1 * r1 * (particle.best_pos[i] - particle.pos[i]);
                    let social_component = c2 * r2 * (swarm.best_pos[i] - particle.pos[i]);

                    let mut new_velocity =
                        particle.velocity[i] + cognitive_component + social_component;

                    // Aplica o fator de constrição k
                    new_velocity *= k;

                    // Verifica se a velocidade excede o máximo
                    if new_velocity > V_MAX {
                        new_velocity = V_MAX;
                    } else if new_velocity < -V_MAX {
                        new_velocity = -V_MAX;
                    }
                    particle.velocity[i] = new_velocity;
                }

                // Atualiza a posição atual da partícula
                for i in 0..DIMENSIONS {
                    particle.pos[i] += particle.velocity[i];

                    // Verifica se a partícula está dentro dos limites
                    if particle.pos[i] > B_HI || particle.pos[i] < B_LO {
                        particle.pos[i] = rng.gen_range(B_LO..B_HI);
                    }
                }
                particle.pos_z = cost_function(&particle.pos);

                // Atualiza a melhor posição conhecida da partícula
                if particle.pos_z < particle.best_pos_z {
                    particle.best_pos = particle.pos;
                    particle.best_pos_z = particle.pos_z;
                }

                // Retorna a partícula para possível atualização do melhor global
                particle.clone()
            })
            // Encontra a melhor partícula desta iteração
            .min_by(|a, b| a.pos_z.partial_cmp(&b.pos_z).unwrap())
            .unwrap();

        // Atualiza a melhor posição global se necessário
        if best_particle.pos_z < swarm.best_pos_z {
            swarm.best_pos = best_particle.pos;
            swarm.best_pos_z = best_particle.pos_z;
        }

        // Calcula a média dos valores de pos_z das partículas nesta iteração
        let sum_pos_z: f64 = swarm.particles.par_iter().map(|p| p.pos_z).sum();
        let average_pos_z = sum_pos_z / POPULATION as f64;

        // Armazena os dados desta iteração
        iterations_data.push(Iteration {
            iteration_number: curr_iter,
            best_value: swarm.best_pos_z,
            average_value: average_pos_z,
        });

        // Verifica a convergência
        if (swarm.best_pos_z - GLOBAL_BEST).abs() < CONVERGENCE {
            println!(
                "O enxame atingiu o critério de convergência após {} iterações (k).",
                curr_iter
            );
            break;
        }
        curr_iter += 1;
    }

    println!("Melhor posição encontrada (k): {:?}", swarm.best_pos);
    println!("Melhor valor encontrado (k): {}", swarm.best_pos_z);

    // Salva o melhor vetor de posição em um arquivo
    let mut file: File =
        File::create("best_position_k.txt").expect("Não foi possível criar o arquivo");
    writeln!(file, "{:?}", swarm.best_pos).expect("Não foi possível escrever no arquivo");

    // Retorna o vetor com as informações de cada iteração
    iterations_data
}

fn plot_iterations(
    filename: &str,
    data: Vec<Iteration>,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ordena os dados com base no número de iteração
    let mut data: Vec<Iteration> = data;
    data.sort_by_key(|iter| iter.iteration_number);

    // Extrai os valores para os eixos X e Y
    let x_vals: Vec<usize> = data.iter().map(|iter| iter.iteration_number).collect();
    let best_vals: Vec<f64> = data.iter().map(|iter| iter.best_value).collect();
    let avg_vals: Vec<f64> = data.iter().map(|iter| iter.average_value).collect();

    // Adiciona uma pequena constante para evitar log de zero ou valores negativos
    let epsilon = 1e-10;

    // Aplica o logaritmo natural aos valores
    let best_vals_ln: Vec<f64> = best_vals.iter().map(|&val| (val + epsilon).ln()).collect();
    let avg_vals_ln: Vec<f64> = avg_vals.iter().map(|&val| (val + epsilon).ln()).collect();

    // Define os limites dos eixos
    let x_min: usize = *x_vals.first().unwrap_or(&0);
    let x_max: usize = *x_vals.last().unwrap_or(&0);

    let y_min = best_vals_ln
        .iter()
        .chain(avg_vals_ln.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = best_vals_ln
        .iter()
        .chain(avg_vals_ln.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Cria a área de desenho
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Configura o gráfico
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Iteração")
        .y_desc("ln(Valor)")
        .draw()?;

    // Desenha a linha do best_value em vermelho
    chart
        .draw_series(LineSeries::new(
            x_vals
                .iter()
                .zip(best_vals_ln.iter())
                .map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Melhor Valor")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Desenha a linha do average_value em azul
    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(avg_vals_ln.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Valor Médio")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Configura a legenda
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() {
    // Executa a versão base do PSO
    let iterations_data_base: Vec<Iteration> = particle_swarm_optimization_base();

    if let Err(e) = plot_iterations(
        "arquivo_base.png",
        iterations_data_base,
        "PSO Base - Valores ao Longo das Iterações",
    ) {
        eprintln!("Erro ao plotar iterações (Base): {}", e);
    }

    // Executa o PSO com Peso de Inércia w
    let w = 0.729; // Valor de exemplo para w
    let iterations_data_w: Vec<Iteration> = particle_swarm_optimization_w(w);

    if let Err(e) = plot_iterations(
        "arquivo_w.png",
        iterations_data_w,
        "PSO com Peso de Inércia (w) - Valores ao Longo das Iterações",
    ) {
        eprintln!("Erro ao plotar iterações (w): {}", e);
    }

    // Executa o PSO com Fator de Constrição k
    let iterations_data_k: Vec<Iteration> = particle_swarm_optimization_k();

    if let Err(e) = plot_iterations(
        "arquivo_k.png",
        iterations_data_k,
        "PSO com Fator de Constrição (k) - Valores ao Longo das Iterações",
    ) {
        eprintln!("Erro ao plotar iterações (k): {}", e);
    }
}

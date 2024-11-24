use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

use std::f64::consts::PI;

// Constantes
const DIMENSIONS: usize = 10; // Altere para o número de dimensões desejado
const GLOBAL_BEST: f64 = 0.0; // Melhor valor global da função de custo
const B_LO: f64 = -5.0; // Limite inferior do espaço de busca
const B_HI: f64 = 5.0; // Limite superior do espaço de busca

const POPULATION: usize = 20; // Número de partículas no enxame
const V_MAX: f64 = 0.1; // Velocidade máxima
const PERSONAL_C: f64 = 2.0; // Coeficiente pessoal
const SOCIAL_C: f64 = 2.0; // Coeficiente social
const CONVERGENCE: f64 = 0.001; // Critério de convergência
const MAX_ITER: usize = 100; // Número máximo de iterações

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
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    let sum_sq: f64 = pos.iter().map(|&xi| xi.powi(2)).sum();
    let sum_cos: f64 = pos.iter().map(|&xi| (c * xi).cos()).sum();

    let term_1 = (-b * (sum_sq / DIMENSIONS as f64).sqrt()).exp();
    let term_2 = (sum_cos / DIMENSIONS as f64).exp();

    -a * term_1 - term_2 + a + 1.0_f64.exp()
}

// Função principal do algoritmo PSO
fn particle_swarm_optimization() -> Vec<Iteration> {
    // Inicializa o enxame
    let mut swarm = Swarm::new(POPULATION, V_MAX);

    // Inicializa o peso de inércia
    let mut rng = rand::thread_rng();
    let inertia_weight = 0.5 + (rng.gen_range(0.0..1.0) / 2.0);

    let mut curr_iter = 0;

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

                    // Atualiza a velocidade da partícula
                    let personal_coefficient =
                        PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i]);
                    let social_coefficient = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i]);
                    let mut new_velocity = inertia_weight * particle.velocity[i]
                        + personal_coefficient
                        + social_coefficient;

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
                "O enxame atingiu o critério de convergência após {} iterações.",
                curr_iter
            );
            break;
        }
        curr_iter += 1;
    }

    println!("Melhor posição encontrada: {:?}", swarm.best_pos);
    println!("Melhor valor encontrado: {}", swarm.best_pos_z);

    // Retorna o vetor com as informações de cada iteração
    iterations_data
}

fn plot_iterations(filename: &str, data: Vec<Iteration>) -> Result<(), Box<dyn std::error::Error>> {
    // Ordena os dados com base no número de iteração
    let mut data = data;
    data.sort_by_key(|iter| iter.iteration_number);

    // Extrai os valores para os eixos X e Y
    let x_vals: Vec<usize> = data.iter().map(|iter| iter.iteration_number).collect();
    let best_vals: Vec<f64> = data.iter().map(|iter| iter.best_value).collect();
    let avg_vals: Vec<f64> = data.iter().map(|iter| iter.average_value).collect();

    // Define os limites dos eixos
    let x_min = *x_vals.first().unwrap_or(&0);
    let x_max = *x_vals.last().unwrap_or(&0);
    let y_min = best_vals
        .iter()
        .chain(avg_vals.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let y_max = best_vals
        .iter()
        .chain(avg_vals.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Cria a área de desenho
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Configura o gráfico
    let mut chart = ChartBuilder::on(&root)
        .caption("Valores ao Longo das Iterações", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Iteração")
        .y_desc("Valor")
        .draw()?;

    // Desenha a linha do best_value em vermelho
    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(best_vals.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Melhor Valor")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Desenha a linha do average_value em azul
    chart
        .draw_series(LineSeries::new(
            x_vals.iter().zip(avg_vals.iter()).map(|(&x, &y)| (x, y)),
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
    let iterations_data = particle_swarm_optimization();

    if let Err(e) = plot_iterations("arquivo.png", iterations_data) {
        eprintln!("Error plotting iterations: {}", e);
    }
}

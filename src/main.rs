use rand::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

// Constantes
const DIMENSIONS: usize = 2; // Número de dimensões
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

// Implementação das funções para o Enxame
impl Swarm {
    fn new(population: usize, v_max: f64) -> Swarm {
        let mut particles = Vec::new();
        let mut best_pos = [0.0; DIMENSIONS];
        let mut best_pos_z = std::f64::INFINITY;
        let mut rng = rand::thread_rng();

        for _ in 0..population {
            let x = rng.gen_range(B_LO..B_HI);
            let y = rng.gen_range(B_LO..B_HI);
            let z = cost_function(x, y);
            let velocity = [rng.gen_range(0.0..v_max), rng.gen_range(0.0..v_max)];
            let particle = Particle {
                pos: [x, y],
                pos_z: z,
                velocity,
                best_pos: [x, y],
                best_pos_z: z,
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

// Função de custo (Ackley)
fn cost_function(x: f64, y: f64) -> f64 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    let term_1 = (-b * ((0.5 * (x.powi(2) + y.powi(2))).sqrt())).exp();
    let term_2 = ((c * x).cos() + (c * y).cos()) / 2.0;
    let term_2 = term_2.exp();

    -a * term_1 - term_2 + a + 1.0_f64.exp()
}

// Função principal do algoritmo PSO
fn particle_swarm_optimization() {
    // Inicializa o enxame
    let mut swarm = Swarm::new(POPULATION, V_MAX);

    // Inicializa o peso de inércia
    let mut rng = rand::thread_rng();
    let inertia_weight = 0.5 + (rng.gen_range(0.0..1.0) / 2.0);

    let mut curr_iter = 0;

    while curr_iter < MAX_ITER {
        // Paraleliza a atualização das partículas
        let best_particle = swarm
            .particles
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
                particle.pos_z = cost_function(particle.pos[0], particle.pos[1]);

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
}

fn main() {
    particle_swarm_optimization();
}

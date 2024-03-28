"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    np.random.seed(42)

    imgs_dir = os.path.join(input_dir, "TUD-Campus")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list = sorted(imgs_list)

    save_frames = {28: os.path.join(output_dir, 'ps5-5-a-1.png'),
                   56: os.path.join(output_dir, 'ps5-5-a-2.png'),
                   71: os.path.join(output_dir, 'ps5-5-a-3.png')}
    t1 = {
        'x': 80,
        'y': 150,
        'w': 60,
        'h': 250,
        'start_frame':1,
    }
    t2 = {
        'x': 140,
        'y': 180,
        'w': 70,
        'h': 160,
        'start_frame':24,
        'end_frame':35
    }

    t3 = {
        'x': 0,
        'y': 173,
        'w': 60,
        'h': 210,
        'start_frame':25
    }

    kwargs1 = {
        'num_particles': 1000,
        'sigma_exp': 4,
        'sigma_dyn': 17,
        'alpha': 0.3,
    }

    kwargs2 = {
        'num_particles': 1000,
        'sigma_exp': 4,
        'sigma_dyn': 16,
        'alpha': 0.1,
    }

    kwargs3 = {
        'num_particles': 1000,
        'sigma_exp': 4,
        'sigma_dyn': 18,
        'alpha': 0.5,
    }



    # Initialize objects
    pf1 = None
    pf2 = None
    pf3 = None

    frame_id = 0
    for img in imgs_list:
        frame = cv2.imread(os.path.join(os.path.join(input_dir, "TUD-Campus"), img))
        frame_id += 1
        if frame_id == t1['start_frame']:
            template1 = frame[int(t1['y']): int(t1['y'] + t1['h']), int(t1['x']): int(t1['x'] + t1['w'])]
            pf1 = ps5.AppearanceModelPF(frame, template=template1, template_coords=t1, **kwargs1)
        if frame_id == t2['start_frame']:
            template2 = frame[int(t2['y']):int(t2['y'] + t2['h']), int(t2['x']): int(t2['x'] + t2['w'])]
            pf2 = ps5.AppearanceModelPF(frame, template=template2, template_coords=t2, **kwargs2)
        if frame_id == t3['start_frame']:
            template3 = frame[int(t3['y']):int(t3['y'] + t3['h']), int(t3['x']): int(t3['x'] + t3['w'])]
            pf3 = ps5.AppearanceModelPF(frame, template=template3, template_coords=t3, **kwargs3)

        # Process frame
        if frame_id >= t1['start_frame']:
            pf1.process(frame)
        if (frame_id >= t2['start_frame']) & (frame_id <= t2['end_frame']):
            pf2.process(frame)

        if frame_id >= t3['start_frame']:
            pf3.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            if frame_id >= t1['start_frame']:
                pf1.render(out_frame)
            if (frame_id >= t2['start_frame']) & (frame_id <= t2['end_frame']):
                pf2.render(out_frame)
            if frame_id >=t3['start_frame']:
                pf3.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_id in save_frames:
            frame_out = frame.copy()
            if frame_id >= t1['start_frame']:
                pf1.render(frame_out)
            if (frame_id >= t2['start_frame']) & (frame_id <= t2['end_frame']):
                pf2.render(frame_out)
            if frame_id >= t3['start_frame']:
                pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_id], frame_out)


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    np.random.seed(42)
    imgs_dir = os.path.join(input_dir, "follow")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]

    imgs_list = sorted(imgs_list)


    save_frames = {60: os.path.join(output_dir, 'ps5-6-a-1.png'),
                   160: os.path.join(output_dir, 'ps5-6-a-2.png'),
                   186: os.path.join(output_dir, 'ps5-6-a-3.png')}
    t1 = {
        'x': 90,
        'y': 40,
        'w': 80,
        'h': 200,
        'start_frame': 1
    }
    t2 = {
        'x': 180,
        'y': 0,
        'w': 90,
        'h': 180,
        'start_frame': 34
    }

    t3 = {
        'x': 170,
        'y': 50,
        'w': 70,
        'h': 165,
        'start_frame': 98
    }

    kwargs1 = {
        'template_min_h': 70,
        'num_particles': 400,
        'sigma_exp': 4,
        'sigma_dyn': 15,
        'alpha': 0.3,
        'max_mse': 4100,
        'scale': 1,
        'scale_rate': 1.05
    }

    kwargs2 = {
        'template_min_h': 70,
        'num_particles': 400,
        'sigma_exp': 4,
        'sigma_dyn': 15,
        'alpha': 0.3,
        'max_mse':4100,
        'scale':1,
        'scale_rate':0.999
    }

    kwargs3 = {
        'template_min_h': 70,
        'num_particles': 500,
        'sigma_exp': 5,
        'sigma_dyn': 10,
        'alpha': 0.2,
        'max_mse': 4100,
        'scale': 1,
        'scale_rate': 1
    }





    # Initialize objects
    pf1 = None

    frame_id = 0
    for img in imgs_list:
        frame = cv2.imread(os.path.join(os.path.join(input_dir, "follow"), img))
        frame_id += 1
        if frame_id == t1['start_frame']:
            template1 = frame[int(t1['y']): int(t1['y'] + t1['h']), int(t1['x']): int(t1['x'] + t1['w'])]
            pf1 = ps5.MDParticleFilter(frame, template=template1, template_coords=t1, **kwargs1)
        elif frame_id == t2['start_frame']:   #update particle filter
            template2 = frame[int(t2['y']): int(t2['y'] + t2['h']), int(t2['x']): int(t2['x'] + t2['w'])]
            pf2 = ps5.MDParticleFilter(frame, template=template2, template_coords=t2, **kwargs2)
        elif frame_id == t3['start_frame']:  # update particle filter
            template3 = frame[int(t3['y']): int(t3['y'] + t3['h']), int(t3['x']): int(t3['x'] + t3['w'])]
            pf3 = ps5.MDParticleFilter(frame, template=template3, template_coords=t3, **kwargs3)

        # Process frame
        if frame_id < t2['start_frame']:
            pf1.process(frame)
        elif frame_id < t3['start_frame']:
            pf2.process(frame)
        else:
            pf3.process(frame)


        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            if frame_id < t2['start_frame']:
                pf1.render(out_frame)
            elif frame_id < t3['start_frame']:
                pf2.render(out_frame)
            else:
                pf3.render(out_frame)


            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_id in save_frames:
            frame_out = frame.copy()
            if frame_id < t2['start_frame']:
                pf1.render(frame_out)
            elif frame_id < t3['start_frame']:
                pf2.render(frame_out)
            else:
                pf3.render(frame_out)

            cv2.imwrite(save_frames[frame_id], frame_out)


if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    # part_5()
    part_6()
